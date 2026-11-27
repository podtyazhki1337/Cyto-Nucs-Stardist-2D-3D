# CytoNucs in one image: training, inference, scoring, the dataset converters and the paper's analyses
# (TensorFlow 2.18 with the CUDA libraries of its pip wheels; the host needs only the NVIDIA driver and the
# NVIDIA Container Toolkit).
#
#   docker build -t podtyazhki1337/cytonucs .      (or: docker pull podtyazhki1337/cytonucs)
#   docker run --rm podtyazhki1337/cytonucs help
#
# docker/run.sh runs the image with the data/, runs/ and models/ folders of the current folder mounted;
# see "Docker" in README.md.
FROM python:3.10-slim-bookworm

# libGL and GLib for OpenCV
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# The versions of the tested environment (docker/constraints.txt) within the ranges of requirements.txt
COPY requirements.txt docker/constraints.txt /tmp/build/
RUN pip install --no-cache-dir -r /tmp/build/requirements.txt -c /tmp/build/constraints.txt \
 && pip install --no-cache-dir -c /tmp/build/constraints.txt nibabel setuptools \
 && pip install --no-cache-dir --no-deps -c /tmp/build/constraints.txt edt \
 && rm -rf /tmp/build

ENV PYTHONPATH=/opt/cytonucs \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLBACKEND=Agg \
    HOME=/tmp \
    PATH=/usr/local/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH

WORKDIR /opt/cytonucs
COPY . /opt/cytonucs
RUN install -m 755 docker/cytonucs /usr/local/bin/cytonucs \
 && mkdir -p -m 777 data runs models

ENTRYPOINT ["cytonucs"]
CMD ["help"]
