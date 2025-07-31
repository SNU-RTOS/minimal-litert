#!/usr/bin/env python3
"""export_tflite_from_timm_v3.py

Unified exporter that converts TIMM ImageNet models to multiple TFLite
variants (FP32, FP16, Dynamic‑range INT8, Full INT8, PT2E INT8) and
verifies them against the original PyTorch network.

Key changes vs. the original three scripts
──────────────────────────────────────────
* Preserves the **original constant‑assignment style** for the configuration
  block (no SimpleNamespace).
* Splits quantized inference helper logic so **int8** and **uint8** tensors
  are handled by distinct code paths, mirroring TensorFlow‑Lite semantics.
"""
from __future__ import annotations

import os
import sys
import random
import logging
from pathlib import Path
from typing import Callable, List
import warnings
import glob
import numpy as np
import requests
import torch
import timm
from PIL import Image
from timm.utils.model import reparameterize_model
from timm.data import resolve_data_config, create_transform

# ---------

from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e

import ai_edge_torch
from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer
from ai_edge_torch.generative.quantize.quant_recipes import full_int8_dynamic_recipe

import ai_edge_quantizer
from ai_edge_litert.interpreter import Interpreter, OpResolverType, Delegate
from ai_edge_litert.interpreter import InterpreterWithCustomOps

from tensorflow.lite.python.lite import Optimize, OpsSet
from tensorflow.python.framework import dtypes as tf_dtypes

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_VAL_DIR = "./dataset/imagenet-1k/val"
EXPORT_ROOT = "./exported"  # Directory to save exported models
SAMPLE_IMAGE_DIR = "./val_imgs/"  # Directory containing sample images for verification
MODEL_LIST_TXT = "./config/model_selected_list.txt"
REP_SAMPLES = 500  # representative images per whole ImageNet val
LOG_FILE = "./log/export_tflite_from_timm_v3.log"
GPU_DELEGATE_PATH = "./lib/libtensorflowlite_gpu_delegate.so"  # Path to EdgeTPU delegate
# GPU_DELEGATE_PATH = None  # Set to None to disable GPU delegate

# ─────────────────────────────────────────────────────────────────────────────
# Seeds & environment
# ─────────────────────────────────────────────────────────────────────────────
SEED = 128
np.random.seed(SEED)
random.seed(SEED)
# os.environ.setdefault("PJRT_DEVICE", "CPU")
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

device = torch.device("cpu")  # Use CPU for inference to avoid GPU dependencies

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
log_file = open(LOG_FILE, "w", buffering=1)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger("export")

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────


def load_model(name: str):
    model = timm.create_model(name, pretrained=True).eval()
    cfg = resolve_data_config(model.pretrained_cfg)
    tfm = create_transform(**cfg)
    return model, tfm, cfg


def build_representative_dataset_tflite(
    val_dir: str, tfm: Callable, total: int = REP_SAMPLES
):
    class_dirs = [d for d in sorted(glob.glob(f"{val_dir}/*")) if os.path.isdir(d)]
    per_cls = max(1, total // len(class_dirs))
    paths: List[str] = []
    for d in class_dirs:
        imgs = sorted(glob.glob(os.path.join(d, "*.JPEG")))
        paths += random.sample(imgs, per_cls) if len(imgs) >= per_cls else imgs
    random.shuffle(paths)

    def gen():
        for p in paths:
            arr = tfm(Image.open(p).convert("RGB")).numpy().transpose(1, 2, 0)
            yield [np.expand_dims(arr.astype(np.float32), 0)]

    return gen


def build_representative_dataset_pt2e(
    val_dir: str, tfm: Callable, total: int = REP_SAMPLES
) -> List[torch.Tensor]:
    class_dirs = [d for d in sorted(glob.glob(f"{val_dir}/*")) if os.path.isdir(d)]
    per_cls = max(1, total // len(class_dirs))
    paths: List[str] = []
    for d in class_dirs:
        imgs = sorted(glob.glob(os.path.join(d, "*.JPEG")))
        paths += random.sample(imgs, per_cls) if len(imgs) >= per_cls else imgs
    random.shuffle(paths)

    tensors = [
        torch.from_numpy(tfm(Image.open(p).convert("RGB")).numpy().transpose(1, 2, 0))
        .unsqueeze(0)
        .float()
        .to(device)
        for p in paths[:total]
    ]
    return tensors


def preprocess(path: str, tfm: Callable) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.expand_dims(tfm(img).numpy(), 0)


def topk(logits: np.ndarray, k: int = 5):
    probs = torch.softmax(torch.tensor(logits.astype(np.float32))[0], 0)
    v, i = torch.topk(probs, k)
    labels = (
        requests.get(
            "https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt"
        )
        .text.strip()
        .split("\n")
    )
    try:
        return [{"label": labels[idx], "value": val.item()} for val, idx in zip(v, i)]
    except IndexError:
        log.warning("topk parsing failed (out of range), returning empty list")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Quantized I/O helpers – int8  vs  uint8 split
# ─────────────────────────────────────────────────────────────────────────────


def _quantize_int8(img: np.ndarray, scale: float, zp: int):
    q = np.clip(np.round(img / scale + zp), -128, 127).astype(np.int8)
    return q


def _quantize_uint8(img: np.ndarray, scale: float, zp: int):
    q = np.clip(np.round(img / scale + zp), 0, 255).astype(np.uint8)
    return q


def _dequant(q: np.ndarray, scale: float, zp: int) -> np.ndarray:
    return (q.astype(np.float32) - zp) * scale if scale != 0 else q.astype(np.float32)


def run_litert(path: str, np_nchw: np.ndarray) -> np.ndarray:
    """Execute a TFLite model and return *de‑quantized* logits."""
    
    from ai_edge_litert.interpreter import load_delegate
    
    
    if GPU_DELEGATE_PATH is not None:
        gpu_delegate = load_delegate(GPU_DELEGATE_PATH)
        intr = Interpreter(
            model_path=path,
            experimental_delegates=[gpu_delegate])
    else:   
        intr = Interpreter(
            model_path=path,
            experimental_op_resolver_type=OpResolverType.AUTO,  # Use XNNPACK Delegate
            num_threads=8,
        )
    # else:
    #     intr = Interpreter(
    #         model_path=path,
    #         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES, # Use built-in ops only
    #         num_threads=8,
    #     )
    
    intr.allocate_tensors()

    # ── Input ──
    inp = intr.get_input_details()[0]
    nhwc = np_nchw.transpose(0, 2, 3, 1)  # NCHW→NHWC
    scale_in, zp_in = inp["quantization"]
    if inp["dtype"] == np.float32:
        payload = nhwc.astype(np.float32)
    elif inp["dtype"] == np.int8:
        payload = _quantize_int8(nhwc, scale_in, zp_in)
    elif inp["dtype"] == np.uint8:
        payload = _quantize_uint8(nhwc, scale_in, zp_in)
    else:
        raise TypeError(f"Unsupported input dtype: {inp['dtype']}")
    intr.set_tensor(inp["index"], payload)

    # ── Invoke ──
    intr.invoke()

    # ── Output ──
    out = intr.get_output_details()[0]
    q_out = intr.get_tensor(out["index"])

    if out["dtype"] in (np.int8, np.uint8):
        scale_out, zp_out = out["quantization"]
        logits = _dequant(q_out, scale_out, zp_out)
    else:  # already fp32 / fp16
        logits = q_out.astype(np.float32)

    return logits


# ─────────────────────────────────────────────────────────────────────────────
# PT2E INT8 export helper (lazy heavy imports)
# ─────────────────────────────────────────────────────────────────────────────


def export_pt2e_int8(
    torch_model: torch.nn.Module, calibration_inputs: List[torch.Tensor], dest: Path
):

    captured_graph = capture_pre_autograd_graph(
        torch_model.eval(), (calibration_inputs[0],)
    )
    graph = prepare_pt2e(captured_graph, PT2EQuantizer())
    for sample in calibration_inputs:
        graph(sample)
    graph = convert_pt2e(graph, fold_quantize=True)

    pt2e_edge = ai_edge_torch.convert(
        graph, (sample,), quant_config=full_int8_dynamic_recipe()
    )
    pt2e_edge.export(str(dest))


# ─────────────────────────────────────────────────────────────────────────────
# Export routine per model
# ─────────────────────────────────────────────────────────────────────────────


def export_variants(name: str):
    log.info("\u25b6 %s", name)
    model_dir = Path(EXPORT_ROOT) / name
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_dir.is_dir():
        model, tfm, _ = load_model(name)
        model = reparameterize_model(model).to(device)

        h, w = tfm.transforms[1].size
        sample_chw = torch.randn(1, 3, h, w, device=device)
        sample_nhwc = sample_chw.permute(0, 2, 3, 1)

        nhwc_mod = ai_edge_torch.to_channel_last_io(model, args=[0]).eval()

        # file paths
        paths = {
            "FP32": model_dir / f"{name}.tflite",
            "FP16": model_dir / f"{name}_fp16.tflite",
            "DYN": model_dir / f"{name}_dyn.tflite",
            "INT8": model_dir / f"{name}_int8.tflite",
            "PT2E": model_dir / f"{name}_pt2e_int8.tflite",
        }

        # FP32
        if not paths["FP32"].exists():
            log.info("  Exporting FP32 model to %s", paths["FP32"])
            edge = ai_edge_torch.convert(
                nhwc_mod,
                (sample_nhwc,),
                _saved_model_dir=str(model_dir),
                _ai_edge_converter_flags={
                    "experimental_enable_resource_variables": True,
                },
            )
            edge.export(str(paths["FP32"]))

        # # Dynamic INT8
        # if not paths["DYN"].exists():
        #     log.info("  Exporting Dynamic INT8 model to %s", paths["DYN"])
        #     edge = ai_edge_torch.convert(
        #         nhwc_mod,
        #         (sample_nhwc,),
        #         _ai_edge_converter_flags={
        #             "optimizations": [Optimize.DEFAULT],
        #             "experimental_enable_resource_variables": True,
        #         },
        #     )
        #     edge.export(str(paths["DYN"]))

        # # FP16
        # if not paths["FP16"].exists():
        #     log.info("  Exporting FP16 model to %s", paths["FP16"])
        #     edge = ai_edge_torch.convert(
        #         nhwc_mod,
        #         (sample_nhwc,),
        #         _ai_edge_converter_flags={
        #             "optimizations": [Optimize.DEFAULT],
        #             "target_spec.supported_types": [tf_dtypes.float16],
        #             "experimental_enable_resource_variables": True,
        #         },
        #     )
        #     edge.export(str(paths["FP16"]))

        # # Full INT8
        # # Note: This is a full INT8 model, not dynamic range quantization.
        # if not paths["INT8"].exists():
        #     log.info("  Exporting Full INT8 model to %s", paths["INT8"])
        #     edge = ai_edge_torch.convert(
        #         nhwc_mod,
        #         (sample_nhwc,),
        #         _ai_edge_converter_flags={
        #             "optimizations": [Optimize.DEFAULT],
        #             "representative_dataset": build_representative_dataset_tflite(
        #                 IMAGENET_VAL_DIR, tfm, REP_SAMPLES
        #             ),
        #             "target_spec.supported_ops": [OpsSet.TFLITE_BUILTINS_INT8],
        #             "inference_input_type": tf_dtypes.int8,
        #             "inference_output_type": tf_dtypes.int8,
        #             "target_spec.supported_types": [tf_dtypes.int8],
        #             "experimental_enable_resource_variables": True,
        #         },
        #     )
        #     edge.export(str(paths["INT8"]))

        # PT2E INT8
        if not paths["PT2E"].exists():
            try:
                log.info("  Exporting PT2E INT8 model to %s", paths["PT2E"])
                pt2e_inputs = build_representative_dataset_pt2e(
                    IMAGENET_VAL_DIR, tfm, total=100
                )
                export_pt2e_int8(nhwc_mod, pt2e_inputs, paths["PT2E"])
            except Exception as e:
                log.warning("  ! PT2E export failed: %s", e)

    return model, tfm, paths


# ─────────────────────────────────────────────────────────────────────────────
# Verification per model
# ─────────────────────────────────────────────────────────────────────────────


def verify(
    model: torch.nn.Module, tfm, model_paths, image_dir, k: int = 5, atol: float = 1e-1
):
    img_dir = os.path.dirname(image_dir)
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.[jJpP][pPnN]*")))
    total = len(img_paths)
    log.info("-" * 50)
    log.info("Verify on %d images under %s", total, img_dir)

    for idx, img_path in enumerate(img_paths, 1):
        tag = f"[{idx}/{total}] {os.path.basename(img_path)}"

        inp = preprocess(img_path, tfm)
        model.eval()
        pt_logits = model(torch.from_numpy(inp).float()).detach().cpu()
        if isinstance(pt_logits, torch._subclasses.FakeTensor):
            pt_logits = pt_logits.to(torch.float32).to("cpu")  # 실제 텐서로 변환
        pt_logits = pt_logits.numpy()  # NumPy 배열로 변환
        log.info("%s  PyTorch top-%d: %s", tag, k, topk(pt_logits, k))

        for label, path in model_paths.items():
            if not path.exists():
                continue
            try:
                logits = run_litert(str(path), inp)
            except Exception as e:
                log.warning("%s  ! %s inference failed: %s", tag, label, e)
                continue

            log.info("%s  %-6s top-%d: %s", tag, label, k, topk(logits, k))
            mae = np.mean(np.abs(pt_logits - logits))
            same = np.allclose(pt_logits, logits, atol=atol)
            log.info("%s          MAE %.4f  %s", tag, mae, "✅" if same else "❌")

    log.info("Verification complete for %s", img_dir)
    log.info("-" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open(MODEL_LIST_TXT) as f:
        model_names = [l.strip() for l in f if l.strip()]

    logging.captureWarnings(True)
    # tf.get_logger().setLevel(logging.INFO)

    log.info("Processing %d models…", len(model_names))
    for name in model_names :
        model, tfm, paths = export_variants(name)
        verify(model, tfm, paths, SAMPLE_IMAGE_DIR, k=5, atol=1e-1)
        log.info("%s done.\n", name)

    log.info("All exports complete. Detailed log: %s", LOG_FILE)
