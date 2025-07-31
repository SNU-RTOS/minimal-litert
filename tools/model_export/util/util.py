import torch
from torch import nn
import torch.fx as fx
import torch.nn.functional as F

def replace_interpolate(m : nn.Module):
    for name, module in m.named_children():
        if isinstance(module, nn.Module):  # 재귀적으로 탐색
            replace_interpolate(module)
    if hasattr(m, 'forward'):
        original_forward = m.forward  # 원래 forward 백업

        def new_forward(*args, **kwargs):
            # interpolate를 호출하는 경우 align_corners를 False로 설정
            if torch.nn.functional.interpolate in original_forward.__code__.co_names:
                kwargs['align_corners'] = False
            return original_forward(*args, **kwargs)

        m.forward = new_forward


# 연산자 교체 Transformer 정의
class ReplaceInterpolate(fx.Interpreter):
    def call_function(self, target, args, kwargs):
        if target == F.interpolate:  # interpolate 함수만 감지
            kwargs['align_corners'] = False  # align_corners 수정
        return super().call_function(target, args, kwargs)
    
# 수정된 그래프 적용
class ReplaceInterpolateWithGraph(fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target == F.interpolate:
            kwargs['align_corners'] = False  # align_corners만 False로 설정
        return super().call_function(target, args, kwargs)
