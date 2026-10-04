#!/usr/bin/env bash
# This lab entry point only evaluates deblur on a small original-size sample.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: create .venv and install the dependencies first; see README.md (Lab server)." >&2
  exit 2
fi

# Whitelist options rather than forwarding abbreviations that could enable SR.
args=("$@")
while (( $# )); do
  case "$1" in
    --deblur-model|--input|--limit|--seed|--runs-root)
      if (( $# < 2 )) || [[ "$2" == --* ]]; then
        echo "error: $1 requires a value" >&2
        exit 2
      fi
      shift 2 ;;
    --deblur-model=*|--input=*|--limit=*|--seed=*|--runs-root=*) shift ;;
    --help|-h)
      exec "$PYTHON" evaluation/run_evaluation.py --help ;;
    *)
      echo "error: lab/run.sh only supports deblur; unsupported option: $1" >&2
      exit 2 ;;
  esac
done
set -- "${args[@]}"

# One GPU per run. Preserve an explicitly empty CUDA_VISIBLE_DEVICES so the
# preflight rejects it instead of unexpectedly using a GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

"$PYTHON" - <<'PY'
import os
import torch

print(f"torch: {torch.__version__}; runtime CUDA: {torch.version.cuda}", flush=True)
print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is unavailable; fix GPU access before running evaluation.")
print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
free, total = torch.cuda.mem_get_info()
print(f"VRAM free/total bytes: {free}/{total}", flush=True)
from pathlib import Path
print(Path("/proc/meminfo").read_text().splitlines()[:3], flush=True)
for name in ("memory.max", "memory.current"):
    path = Path("/sys/fs/cgroup") / name
    if path.exists():
        print(f"cgroup {name}: {path.read_text().strip()}", flush=True)
torch.ones(1, device="cuda").sum().item()
PY

"$PYTHON" -m pip check

# Later arguments may override sampling/checkpoint defaults, never the mode.
exec "$PYTHON" evaluation/run_evaluation.py \
  --deblur --input "$ROOT/evaluation/data/input" --limit 1 "$@"
