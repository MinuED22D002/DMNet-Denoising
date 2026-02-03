FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

# --------------------
# Base system
# --------------------
RUN apt-get update && apt-get install -y \
    sudo \
    software-properties-common \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    pkg-config \
    ca-certificates \
    libx11-6 \
    libgl1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    && rm -rf /var/lib/apt/lists/*

# --------------------
# Modern CMake (REQUIRED)
# --------------------
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh \
    && chmod +x cmake-3.27.9-linux-x86_64.sh \
    && ./cmake-3.27.9-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.27.9-linux-x86_64.sh

# --------------------
# Python 3.9
# --------------------
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# --------------------
# pip
# --------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
RUN python3.9 -m pip install --upgrade pip

# --------------------
# Python deps
# --------------------
RUN python3.9 -m pip install torch torchvision torchaudio pyyaml numpy trimesh typing_extensions --upgrade

# --------------------
# Open3D 0.16
# --------------------
WORKDIR /opt
RUN git clone --branch v0.16.0 --depth 1 https://github.com/isl-org/Open3D.git

WORKDIR /opt/Open3D
RUN bash util/install_deps_ubuntu.sh assume-yes

RUN mkdir build
WORKDIR /opt/Open3D/build

RUN cmake .. \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_GUI=OFF \
    -DBUILD_WEBRTC=OFF

RUN make -j$(nproc)
RUN make install

# --------------------
# Environment
# --------------------
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONPATH=/usr/local/lib/python3.9/site-packages

# --------------------
# CGAL and dependencies for DMNet
# --------------------
RUN apt-get update && apt-get install -y \
    libcgal-dev \
    libcgal-qt5-dev \
    libgmp-dev \
    libmpfr-dev \
    && rm -rf /var/lib/apt/lists/*

# --------------------
# DMNet
# --------------------
WORKDIR /workspace
RUN git clone https://github.com/CharizardChenZhang/DMNet.git

WORKDIR /workspace/DMNet
CMD ["/bin/bash"]