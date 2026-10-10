FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/models/torch-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3 -m venv /app/.venv
COPY pyproject.toml requirements-wsl.txt ./
COPY evaluation/requirements.txt ./evaluation/requirements.txt
# Same Python 3.12 / CUDA 12.8 wheels and dependency manifests as the lab.
RUN .venv/bin/python -m pip install --no-cache-dir --no-deps \
        'https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl' \
        'https://download.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl' \
    && .venv/bin/python -m pip install --no-cache-dir -r requirements-wsl.txt -r evaluation/requirements.txt
COPY src/ ./src/
COPY evaluation/*.py evaluation/LICENSE-CPBD.txt ./evaluation/
COPY lab/run.sh ./lab/run.sh
RUN .venv/bin/python -m pip install --no-deps --no-build-isolation -e . \
    && .venv/bin/python -m pip check

# Mount existing models/ and 923海上正攝_lab_extract/ read-only at their /app paths.
# models/ includes the six SR checkpoints, deblur/ and the prepared torch-cache/.
# Mount /app/evaluation/runs writable to retain images and results.csv.
# GPU execution requires a host with NVIDIA Container Toolkit and --gpus.
# --dry-run checks mounted inputs/weights without requiring GPU access.
# Default: 18 frames per height/speed group, 108 originals across six groups.
ENTRYPOINT ["/bin/bash", "lab/run.sh"]
