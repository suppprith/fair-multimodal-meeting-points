# Reproducible environment for fair-multimodal-meeting-points.
# Java 21 is needed for the r5py real-network backend; the data-free Euclidean
# backend and the unit tests run with no data. Large OSM/GTFS inputs are fetched
# separately (see scripts/DATA.md), not baked into the image.
FROM eclipse-temurin:21-jdk-jammy

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt
COPY . .

# Default: run the data-free unit tests.
CMD ["python3", "tests/test_core.py"]
