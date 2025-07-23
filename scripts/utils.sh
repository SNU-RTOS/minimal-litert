#!/bin/bash

# =========================================================================== #
# 1. Logging & Output Helpers                                                 #
# =========================================================================== #

# Color definitions for logging
C_RESET='\033[0m'
C_BLUE='\033[1;34m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[1;33m'
C_RED='\033[0;31m'
C_CYAN='\033[0;36m'

# Helper functions for logging
run()         { echo -e "${C_CYAN}▶${C_RESET} $*"; "$@"; }
ensure_dir()  { [[ -d $1 ]] || run mkdir -p "$1"; }
banner()      { echo -e "\n${C_BLUE}════════════════════════════════════════════════════════════════════════════════${C_RESET}"; \
                echo -e "${C_BLUE} ≫${C_RESET} ${C_YELLOW}$*${C_RESET}"; \
                echo -e "${C_BLUE}════════════════════════════════════════════════════════════════════════════════${C_RESET}"; }
log()         { echo -e "${C_GREEN}✔${C_RESET} $@"; }
warn()        { echo -e "${C_YELLOW}⚠ Warning:${C_RESET} $@"; }
error() {
  echo -e "${C_RED}✖ Error in ${BASH_SOURCE[1]}:${BASH_LINENO[0]}:${C_RESET} \n -> $*" >&2
  exit 1;
}
execute_with_log() {
    # Executes a command, redirecting its output based on the implementation chosen below.
    # To switch behavior, comment out the active implementation and uncomment the desired one.
    # Usage: execute_with_log <log_file_path> <command...>
    local log_file="$1"; shift

    if [[ "${LOG_ENABLED:-false}" == "true" ]]; then
        if [[ -z "$log_file" ]]; then
            error "Log file path not provided to execute_with_log."
            return 1
        fi

        # --- Implementation 1: Log to console AND file (default) ---
        # Output is shown on the console (with color) and appended to the log file (without color).
        # ( "$@" ) 2>&1 | tee >(sed -r "s/\x1b\[[0-9;]*m//g" > "$log_file")

        # --- Implementation 2: Log ONLY to file ---
        # # Output is NOT shown on the console, only appended to the log file (without color).
        ( "$@" ) 2>&1 | sed -r "s/\x1b\[[0-9;]*m//g" > "$log_file"
    else
        "$@"
    fi
}


# =========================================================================== #
# 2. System & Benchmark Helpers                                               #
# =========================================================================== #

clear_caches() {
    banner "Clearing System Caches"
    log "Dropping OS Page Caches..."
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    log "Waiting for caches to clear..."
    sleep 1
    
    log "Dropping swapped memory..."
    sudo swapoff -a
    sudo swapon -a
    sleep 1
    
    log "Clearing CPU Caches..."
    ARCH=$(uname -m)
    
    BIN_DIR="${ROOT_PATH}/tools/bin"
    SRC_DIR="${ROOT_PATH}/tools/cache/src"
    ensure_dir "$BIN_DIR"

    case "$ARCH" in
        "x86_64")
            CACHE_SCRIPT_NAME="clear_cache_x86"
            CACHE_SOURCE="$SRC_DIR/clear_cache_x86.cc"
        ;;
        "aarch64")
            CACHE_SCRIPT_NAME="clear_cache_arm"
            CACHE_SOURCE="$SRC_DIR/clear_cache_arm.cc"
        ;;
        *)
            error "Unsupported architecture: $ARCH"
            return 1
        ;;
    esac

    CACHE_SCRIPT="$BIN_DIR/$CACHE_SCRIPT_NAME"

    if [[ ! -f "$CACHE_SCRIPT" ]]; then
        log "Building cache clearing script for $ARCH..."
        if [[ -f "$CACHE_SOURCE" ]]; then
            g++ -O2 "$CACHE_SOURCE" -o "$CACHE_SCRIPT"
            if [[ $? -ne 0 ]]; then
                error "Failed to build cache clearing script."
            fi
        else
            error "Source file not found: $CACHE_SOURCE"
        fi
    fi
    
    if [[ -f "$CACHE_SCRIPT" ]]; then
        run "$CACHE_SCRIPT"
    else
        log "[WARNING] CPU cache clearing script not found: $CACHE_SCRIPT"
    fi
    
    log "Finished clearing caches."
}

get_pagefault_stats() {
    local pid=$1
    local stat_line=$(cat /proc/$pid/stat 2>/dev/null)
    if [ $? -eq 0 ]; then
        local stats=($stat_line)
        echo "${stats[9]},${stats[11]}"
    else
        echo "0,0"
    fi
}

# =========================================================================== #
# 3. Ftrace Helpers                                                           #
# =========================================================================== #

TRACEFS="/sys/kernel/debug/tracing"
TRACE_SAVE_INTERVAL=5  # Save trace every 5 seconds

setup_ftrace() {
    log "Setting up ftrace..."
    # Clear existing trace
    echo > $TRACEFS/trace
    
    # Disable tracing temporarily
    echo 0 > $TRACEFS/tracing_on
    
    # Clear existing events
    echo > $TRACEFS/set_event
    
    # Enable memory-related events
    echo 1 > $TRACEFS/events/kmem/mm_page_alloc/enable
    echo 1 > $TRACEFS/events/kmem/mm_page_free/enable
    echo 1 > $TRACEFS/events/kmem/rss_stat/enable
    
    # Memory reclaim events
    echo 1 > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_begin/enable
    echo 1 > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_end/enable
    
    # Set the trace buffer size (in KB)
    echo 8192 > $TRACEFS/buffer_size_kb
    
    # Use function tracer
    echo function > $TRACEFS/current_tracer
    
    # Clear the trace buffer
    echo > $TRACEFS/trace
    log "ftrace setup complete."
}

setup_pid_filter() {
    local pid=$1
    log "Setting up ftrace PID filter for PID: $pid"
    echo "common_pid==$pid" > $TRACEFS/events/kmem/mm_page_alloc/filter
    echo "common_pid==$pid" > $TRACEFS/events/kmem/mm_page_free/filter
    echo "common_pid==$pid" > $TRACEFS/events/kmem/rss_stat/filter
    echo "common_pid==$pid" > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_begin/filter
    echo "common_pid==$pid" > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_end/filter
}

save_trace_buffer() {
    local output_file=$1
    local elapsed_time=$2
    
    # Save current trace buffer by appending to the output file
    echo -e "\n# Elapsed time: $elapsed_time seconds" >> "$output_file"
    cat $TRACEFS/trace >> "$output_file"
    
    # Clear the buffer for next collection
    echo > $TRACEFS/trace
    
    log "Appended trace data at $elapsed_time seconds to $output_file"
}


# =========================================================================== #
# 4. Build & Filesystem Helpers                                               #
# =========================================================================== #

# ── Build Configuration ───────────────────────────────────────────────────────
setup_build_config() {
  local BUILD_MODE=${1:-release}
  
  BAZEL_LAUNCH_CONF="--output_user_root=$BAZEL_ROOT"

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
  export BAZEL_CONF COPT_FLAGS LINKOPTS GPU_FLAGS GPU_COPT_FLAGS BAZEL_LAUNCH_CONF
}

create_symlink_or_fail() {
  local src="$1"
  local dst="$2"
  local label="$3"

  if [ ! -e "$src" ]; then
    error "Target not found: $src"
  fi

  log "→ Making symlink: $label"
  ln -sf "$src" "$dst"
}

