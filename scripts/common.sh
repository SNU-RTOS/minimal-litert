# ──────────────────────────────────────────────────────────────────────────────

# ── Build Configuration ───────────────────────────────────────────────────────
setup_build_config() {
  local BUILD_MODE=${1:-release}
  
  if [ "$BUILD_MODE" = "debug" ]; then
    BAZEL_CONF="-c dbg"
    COPT_FLAGS="--copt=-Og"
    LINKOPTS=""
  else
    BAZEL_CONF="-c opt"
    COPT_FLAGS="--copt=-Os --copt=-fPIC "
    LINKOPTS="--linkopt=-s"
  fi

  # GPU Delegate Configuration
  GPU_FLAGS="--define=supports_gpu_delegate=true"
  GPU_COPT_FLAGS="--copt=-DTFLITE_GPU_ENABLE_INVOKE_LOOP=1 --copt=-DCL_DELEGATE_NO_GL --copt=-DTFLITE_SUPPORTS_GPU_DELEGATE=1"
  
  # Export variables for use in calling scripts
  export BAZEL_CONF COPT_FLAGS LINKOPTS GPU_FLAGS GPU_COPT_FLAGS
}

create_symlink_or_fail() {
  local src="$1"
  local dst="$2"
  local label="$3"

  if [ ! -e "$src" ]; then
    echo "❌ Target not found: $src"
    exit 1
  fi

  echo "→ Making symlink: $label"
  ln -sf "$src" "$dst"
}