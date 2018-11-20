FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

LABEL maintainer="sam@myrtle.ai"

#- Upgrade system and install dependencies -------------------------------------
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
                    build-essential \
                    cmake \
                    git \
                    libboost-program-options-dev \
                    libboost-system-dev \
                    libboost-test-dev \
                    libboost-thread-dev \
                    libbz2-dev \
                    libeigen3-dev \
                    liblzma-dev \
                    libsndfile1 \
                    python3 \
                    python3-dev \
                    python3-pip \
                    python3-setuptools \
                    python3-wheel \
                    sudo \
                    vim && \
    rm -rf /var/lib/apt/lists

#- Enable passwordless sudo for users under the "sudo" group -------------------
RUN sed -i.bkp -e \
      's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' \
      /etc/sudoers

#- Create data group for NFS mount --------------------------------------------
RUN groupadd --system data --gid 5555

#- Create and switch to a non-root user ----------------------------------------
RUN groupadd -r ubuntu && \
    useradd --no-log-init \
            --create-home \
            --gid ubuntu \
            ubuntu && \
    usermod -aG sudo ubuntu
USER ubuntu
WORKDIR /home/ubuntu

#- Install Python packages -----------------------------------------------------
ARG WHEELDIR=/home/ubuntu/.cache/pip/wheels
COPY --chown=ubuntu:ubuntu deps deps
COPY --chown=ubuntu:ubuntu requirements.txt requirements.txt
RUN pip3 install --find-links ${WHEELDIR} \
                 -r requirements.txt && \
    rm requirements.txt && \
    rm -r ${WHEELDIR} && \
    rm -r /home/ubuntu/.cache/pip

# warp-ctc
RUN cd /home/ubuntu/deps/warp-ctc/pytorch_binding && \
    git reset --hard 6f3f1cb7871f682e118c49788f5e54468b59c953 && \
    python3 setup.py bdist_wheel && \
    pip3 install dist/warpctc-0.0.0-cp35-cp35m-linux_x86_64.whl

# ctcdecode bindings
RUN cd /home/ubuntu/deps/ctcdecode && \
    git reset --hard 6f4326b43b8dc49fd2e328ce231d1ba37f8e373f && \
    pip3 install .

# kenlm for binaries for building LMs and Python lib for easy analysis
RUN cd /home/ubuntu/deps/kenlm && \
    git reset --hard 328cc2995202e84d29e3773203d29cdd6cc07132 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j 4 && \
    sudo mv bin/lmplz bin/build_binary /usr/bin/ && \
    pip3 install /home/ubuntu/deps/kenlm

RUN sudo rm -rf deps

#- Install deepspeech package --------------------------------------------------
COPY --chown=ubuntu:ubuntu . deepspeech
RUN pip3 install deepspeech/
RUN rm -rf deepspeech

#- Setup Jupyter ---------------------------------------------------------------
EXPOSE 9999
ENV PATH /home/ubuntu/.local/bin:$PATH
ENV SHELL /bin/bash
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0",   \
     "--port=9999"]
