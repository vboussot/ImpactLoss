FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip cmake ninja-build build-essential openmpi-bin libopenmpi-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN git clone https://github.com/InsightSoftwareConsortium/ITK.git && \
    mkdir ITK-build && cd ITK-build && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/install/ITK ../ITK && \
    ninja install && \
    rm -rf /build/ITK /build/ITK-build

RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip && \
    unzip libtorch.zip -d /install && rm libtorch.zip

RUN git clone https://github.com/vboussot/ImpactElastix.git && \
    mkdir elastix-build && cd elastix-build && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DTorch_DIR=/install/libtorch/share/cmake/Torch/ \
          -DITK_DIR=/install/ITK/lib/cmake/ITK-6.0 \
          -DUSE_ImpactMetric=ON \
          -DCMAKE_INSTALL_PREFIX=/install/elastix \
          ../ImpactElastix && \
    ninja install && \
    rm -rf /build/ImpactElastix /build/elastix-build

FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=builder /install/ITK /usr/local/ITK
COPY --from=builder /install/libtorch /usr/local/libtorch
COPY --from=builder /install/elastix /usr/local/elastix

ENV LD_LIBRARY_PATH="/usr/local/libtorch/lib:/usr/local/elastix/lib:${LD_LIBRARY_PATH}"
ENV PATH="/usr/local/elastix/bin:$PATH"

COPY CMD.sh /opt/CMD.sh
RUN chmod +x /opt/CMD.sh

CMD ["/opt/CMD.sh"]