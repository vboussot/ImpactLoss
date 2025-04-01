# syntax = docker/dockerfile:1.2

FROM archlinux

RUN --mount=type=cache,sharing=locked,target=/var/cache/pacman \
    pacman -Syyu --noconfirm --needed base base-devel git wget unzip cmake cuda openmpi

WORKDIR /lib

RUN git clone https://github.com/InsightSoftwareConsortium/ITK.git
RUN mkdir /lib/ITK-build && mkdir /lib/ITK-install
WORKDIR /lib/ITK-build
RUN cmake -D CMAKE_INSTALL_PREFIX=../ITK_install ../ITK
RUN make -j12 install

WORKDIR /lib
RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu124.zip 
RUN unzip libtorch.zip -d . && rm libtorch.zip

ENV PATH="/opt/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/cuda/lib64:${LD_LIBRARY_PATH}"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility


RUN git clone https://github.com/SuperElastix/elastix.git
WORKDIR /lib/elastix
COPY ./Impact.diff ./Impact.diff
RUN git apply ./Impact.diff
RUN mkdir /lib/elastix-build && mkdir /lib/elastix-install
WORKDIR /lib/elastix-build
RUN cmake -DTorch_DIR=../libtorch/share/cmake/Torch/ -DITK_DIR=../ITK_install/lib/cmake/ITK-6.0/ -DCMAKE_INSTALL_PREFIX:PATH=../elastix-install  ../elastix
RUN make -j12 install

WORKDIR /lib
ENV LD_LIBRARY_PATH="/lib/libtorch/lib/:/lib/elastix-install/lib/:${LD_LIBRARY_PATH}"
COPY CMD.sh ./CMD.sh
RUN chmod +x ./CMD.sh
CMD ["./CMD.sh"]