# RTX4080 means sm_89 cuda architecture.
# sm_89 is supported from 11.8 version.
# ref: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# change apt repository to japan server
ARG APT_SERVER=http://ftp.jaist.ac.jp/pub/Linux/ubuntu/
RUN sed -i.bak -e "s%http://[^ ]\+%${APT_SERVER}%g" /etc/apt/sources.list

# Avoid dialog popup in apt command
# ref: https://docs.docker.jp/engine/faq.html#dockerfile-debian-frontend-noninteractive
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    sudo \
    wget \
    git \
    git-lfs \
    vim \
    silversearcher-ag \
    build-essential \
    libopenmpi-dev \
    python3-dev \
    python3-pip \
    p7zip-full \
    && apt clean \
    && rm -Rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# install dockerfile linter
ARG hadolint_bin_url=https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
RUN wget --progress=dot:giga -O /bin/hadolint ${hadolint_bin_url} \
    && chmod a+x /bin/hadolint

ARG username="user0"
RUN useradd --create-home --shell /bin/bash -G sudo,root ${username} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${username}
ENV PATH="/home/${username}/.local/bin:${PATH}"

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib

RUN python -m pip install --no-cache-dir --user --upgrade pip \
    && python -m pip install --no-cache-dir --user --upgrade setuptools wheel \
    && python -m pip install --no-cache-dir --user --upgrade \
        black \
        flake8 \
        python-dateutil \
        python-dotenv \
        fire \
        tqdm \
        transformers \
        accelerate \
        bitsandbytes \
        scipy \
        sentencepiece \
        jupyter \
        datasets \
        peft \
        deepspeed \
        mpi4py \
    && python -m pip install --no-cache-dir --user --upgrade \
        torch --index-url https://download.pytorch.org/whl/cu118

WORKDIR /home/${username}/work
