# syntax = docker/dockerfile:1.2

FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip cmake ninja-build build-essential openmpi-bin libopenmpi-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /lib

RUN git clone https://github.com/InsightSoftwareConsortium/ITK.git
RUN mkdir /lib/ITK-build && mkdir /lib/ITK-install
WORKDIR /lib/ITK-build
RUN cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/lib/ITK-install /lib/ITK \
 && ninja -j"$(nproc)" install

WORKDIR /lib
RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip
RUN unzip libtorch.zip -d . && rm libtorch.zip

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN git clone https://github.com/vboussot/ImpactElastix.git
RUN mkdir -p /lib/elastix-build /lib/elastix-install
WORKDIR /lib/elastix-build

RUN cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DTorch_DIR=/lib/libtorch/share/cmake/Torch/ \
    -DITK_DIR=/lib/ITK-install/lib/cmake/ITK-6.0 \
    -DUSE_ImpactMetric=ON \
    -DCMAKE_INSTALL_PREFIX=/lib/elastix-install \
    /lib/ImpactElastix \
 && ninja -j"$(nproc)" install

WORKDIR /lib
ENV LD_LIBRARY_PATH="/lib/libtorch/lib/:/lib/elastix-install/lib/:${LD_LIBRARY_PATH}"
COPY CMD.sh /lib/CMD.sh
RUN chmod +x /lib/CMD.sh
CMD ["./CMD.sh"]