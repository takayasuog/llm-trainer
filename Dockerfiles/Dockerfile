ARG UBUNTU_VER=20.04

FROM ubuntu:$UBUNTU_VER

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
        sentencepiece \
        jupyter \
        datasets \
        peft \
    && python -m pip install --no-cache-dir --user --upgrade \
        torch --index-url https://download.pytorch.org/whl/cu118

WORKDIR /home/${username}/work
