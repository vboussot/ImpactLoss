FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip cmake ninja-build build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.8.0%2Bcu128.zip && \
    unzip libtorch.zip -d /install && rm libtorch.zip

RUN git clone https://github.com/InsightSoftwareConsortium/ITK.git ITK-source && cd ITK-source && git checkout v5.4.5

RUN mkdir ITK-build && cd ITK-build && \
    cmake -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF \
        -DITK_LEGACY_REMOVE=ON \
        -DCMAKE_INSTALL_PREFIX=/install/ITK \
        -GNinja /build/ITK-source && \
    ninja install && \
    rm -rf /build/ITK-source /build/ITK-build

RUN git clone https://github.com/vboussot/ImpactElastix.git elastix-source

RUN mkdir elastix-build && cd elastix-build && \
    cmake -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DITK_DIR=/install/ITK/lib/cmake/ITK-5.4  \
        -DTorch_DIR=/install/libtorch/share/cmake/Torch/ \
        -DUSE_ALL_COMPONENTS=ON \
        -DBUILD_TESTING:BOOL=OFF \
        -DELASTIX_USE_GTEST=OFF \
        -DCMAKE_INSTALL_PREFIX=/install/elastix \
        -GNinja /build/elastix-source && \
    ninja install && \
    rm -rf /build/elastix-source /build/elastix-build

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 AS runtime

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=builder /install/ITK /usr/local/ITK
COPY --from=builder /install/libtorch /usr/local/libtorch
COPY --from=builder /install/elastix /usr/local/elastix

ENV LD_LIBRARY_PATH="/usr/local/ITK/lib:/usr/local/libtorch/lib:/usr/local/elastix/lib:${LD_LIBRARY_PATH}"
ENV PATH="/usr/local/elastix/bin:$PATH"

COPY CMD.sh /opt/CMD.sh
RUN chmod +x /opt/CMD.sh

CMD ["/opt/CMD.sh"]