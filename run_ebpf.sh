#!/bin/bash
set -e

# Check if the bpftrace script path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_bpftrace_script>"
    exit 1
fi

BPF_SCRIPT_PATH=$1

if [ ! -f "$BPF_SCRIPT_PATH" ]; then
    echo "Error: BPF script not found at '$BPF_SCRIPT_PATH'"
    exit 1
fi

# Execute bpftrace with the provided script file
sudo bpftrace "$BPF_SCRIPT_PATH"