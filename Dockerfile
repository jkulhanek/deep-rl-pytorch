# deepmind-lab package container
FROM kulhanek/deepmindlab as artefact-builder

# Output container
FROM kulhanek/pytorch

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
  lua5.1 \
  liblua5.1-0-dev \
  libffi-dev \
  gettext \
  freeglut3-dev \
  libsdl2-dev \
  libosmesa6-dev \
  realpath \
  build-essential \
  zip \
  git

# Install package
COPY --from=artefact-builder /artifact/DeepMind_Lab-1.0-py3-none-any.whl /tmp/DeepMind_Lab-1.0-py3-none-any.whl

RUN umask 022
RUN pip3 install --upgrade pip

# Tensorflow is needed for baselines
RUN pip3 install tensorflow

RUN pip3 install matplotlib six seaborn visdom gym gym[atari] && \
  pip3 install /tmp/DeepMind_Lab-1.0-py3-none-any.whl && \
  pip3 install git+https://github.com/openai/baselines.git

WORKDIR /root