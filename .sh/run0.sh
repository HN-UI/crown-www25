#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export SCRIPT_TAG="${SCRIPT_TAG:-run0}"
export GPU_ID="${GPU_ID:-0}"

exec bash ./kl_sweep_worker.sh
