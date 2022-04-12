FROM ubuntu:20.04

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

RUN apt-get update -y
# install mivisionx base dependencies - Level 1
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git
# install ROCm for mivisionx OpenCL/HIP dependency - Level 2
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install libnuma-dev wget sudo &&  \
        wget https://repo.radeon.com/amdgpu-install/21.50/ubuntu/focal/amdgpu-install_21.50.50000-1_all.deb && \
        sudo apt-get install -y ./amdgpu-install_21.50.50000-1_all.deb && \
        sudo apt-get update -y && \
        sudo amdgpu-install -y --usecase=rocm

WORKDIR /workspace

# install MIVisionX
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && mkdir build && cd build && \
        cmake -D BACKEND=OCL ../MIVisionX && make -j8 && make install