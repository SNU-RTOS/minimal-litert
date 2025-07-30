# Toolchain Configuration

This document explains how to configure and use the Bazel C++ cross-compilation toolchain defined in this directory.

## 1. Overview

We maintain a generic `cc_toolchain_config` rule that can switch between two implementations:

- **Default**: A generic AArch64 cross-toolchain (`aarch64-none-linux`).
- **S32G**: A custom Yocto-based toolchain (`aarch64-fsl-linux`).

Both are declared by the same rule, selecting the implementation based on the `target_platform` attribute.

## 2. Rule Definition (`tools/cross_compile/BUILD`)

```starlark
cc_toolchain_config(
    name = "<toolchain_name>_config",
    target_platform = "default"  # or "s32g" for the S32G variant
)
```

- `target_platform`: String attribute. Use `"default"` to select the generic AArch64 toolchain or `"s32g"` to select the S32G Yocto toolchain.
- Internally, `_impl_default`, `_impl_s32g` or your customized toolchain function will return a `CcToolchainConfigInfo` with appropriate `tool_paths`, `features`, and flags.

## 3. Quick Customization

To switch or add your own custom toolchain variant, you only need to update the single `cc_toolchain_config` invocation in `tools/cross_compile/BUILD`:

```starlark
// tools/cross_compile/BUILD
cc_toolchain_config(
    name = "<your_toolchain>_config",
    target_platform = "<default|s32g|...>",
)
```

All other rules (`cc_toolchain`, `toolchain` entries) remain unchanged.

### Implementation Example (`cc_toolchain_config.bzl`)

In `cc_toolchain_config.bzl`, the `_impl` function selects the right implementation based on `target_platform`:

```starlark
def _impl(ctx):
    if ctx.attr.target_platform == "s32g":
        return _impl_s32g(ctx)
    else:
        return _impl_default(ctx)

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "target_platform": attr.string(mandatory = False, default = "default"),
    },
    provides = [CcToolchainConfigInfo],
)
```

This logic handles both the generic AArch64 toolchain and the S32G Yocto variant. Just changing the `target_platform` in **BUILD** will switch implementations.

### S32G Implementation Details

The `_impl_s32g` function in `cc_toolchain_config.bzl` provides the custom Yocto-based toolchain for S32G. It includes:

- **tool_paths**: Paths to cross-compiler binaries for AArch64 under the S32G Yocto sysroot.
- **features**:
  - Linker flags (`--sysroot`, `-march`, `-mtune`, etc.) tailored to the S32G platform.
  - Compile flags (`--sysroot`, architecture tuning) for S32G.
- **cxx_builtin_include_directories**: GCC include paths within the Yocto sysroot.
- **toolchain_identifier**: A unique string (e.g. `"aarch64-fsl-linux"`) identifying the S32G toolchain.

Excerpt from `cc_toolchain_config.bzl`:

```starlark
# tools/cross_compile/cc_toolchain_config.bzl

def _impl_s32g(ctx):
    # Define tool_paths for gcc, ld, ar, cpp, etc. under S32G sysroot
    tool_paths = [ ... ]

    # Configure link and compile flags for the S32G platform
    features = [ ... ]

    # Create and return the toolchain config info
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        tool_paths = tool_paths,
        features = features,
        cxx_builtin_include_directories = [ ... ],
        toolchain_identifier = "aarch64-fsl-linux",
        host_system_name = "x86_64",
        target_system_name = "aarch64-fsl-linux",
        target_cpu = "aarch64",
        compiler = "gcc",
        abi_version = "gcc",
        abi_libc_version = "glibc",
        target_libc = "glibc",
    )
```

Bazel will select this implementation when you set like following codes in `tools/cross_compile/BUILD`:

```starlark
cc_toolchain_config(
    name = "fsl_s32g_toolchain_config",
    target_platform = "s32g",
)
```

## 4. Building with Bazel

Select the platform when invoking Bazel:

```bash
./build.sh --cross_compile
```

---

This setup allows easy switching between the generic AArch64 toolchain and the custom S32G Yocto toolchain via a single attribute on the same Starlark rule.
