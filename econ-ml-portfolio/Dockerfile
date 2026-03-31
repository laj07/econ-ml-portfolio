FROM python:3.11-slim

LABEL maintainer="you@youremail.com"
LABEL description="Economics ML Portfolio — reproducible experiments for development economics"

WORKDIR /app

# System dependencies for torch, scipy, rasterio (satellite imagery), GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying source so this layer is cached on re-builds
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# PyG — install after torch so the right CUDA flavour is used.
# For GPU builds, replace +cpu with your CUDA version, e.g. +cu121
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-geometric \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

COPY . .

ENV PYTHONPATH=/app

ENTRYPOINT ["bash", "docker_entrypoint.sh"]
