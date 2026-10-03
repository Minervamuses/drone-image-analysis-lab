FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Keep the same Linux x86_64 / Python 3.12 / CUDA 12.8 wheels as the local setup.
COPY requirements-wsl.txt ./
RUN python -m pip install --no-deps --progress-bar off \
      'https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl' \
      'https://download.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl' \
    && python -m pip install --only-binary=:all: --progress-bar off -r requirements-wsl.txt

COPY pyproject.toml ./
COPY src/drone_sr/ ./src/drone_sr/
# Editable installation preserves MODEL_PATH at /app/models/model.pth.
RUN python -m pip install --no-deps --no-build-isolation -e . \
    && python -m pip check \
    && mkdir -p input output models

# Host UIDs passed with --user may have no passwd entry in the container.
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

ENTRYPOINT ["python", "-m", "drone_sr"]
