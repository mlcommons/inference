FROM ubuntu:18.04

ENV PYTHON=/opt/anaconda3/envs/mlperf-rnnt/bin/python

ENV PATH /opt/anaconda3/bin:$PATH

WORKDIR /root

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
      git \
      build-essential \
      software-properties-common \
      ca-certificates \
      wget \
      curl \
      htop \
      zip \
      unzip\
      libsndfile1

COPY ./env*  ./

RUN cd /opt && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash ./miniconda.sh -b -p /opt/anaconda3 && \
    rm miniconda.sh && \
    /opt/anaconda3/bin/conda clean -tipsy && \
    ln -s /opt/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    conda config --set always_yes yes --set changeps1 no

RUN conda env create -f  ./env_conda.yml

RUN echo "conda activate mlperf-rnnt" >> ~/.bashr

RUN $PYTHON -m  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r env_pip.txt

RUN git clone --recurse-submodules https://github.com/mlcommons/inference

RUN mkdir -p ~/inference/speech_recognition/rnnt/third/install-dir \
    && cd  ~/inference/speech_recognition/rnnt/third \
    && wget https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz \
    && tar xf flac-1.3.2.tar.xz && cd flac-1.3.2 \
    && ./configure --prefix=/root/inference/speech_recognition/rnnt/third/install-dir && make && make install

RUN cd  ~/inference/speech_recognition/rnnt/third \
    && wget https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz \
    && tar zxf sox-14.4.2.tar.gz && cd sox-14.4.2 \
    && ./configure --prefix=/root/inference/speech_recognition/rnnt/third/install-dir && make && make install

RUN cd ~/inference/loadgen/ && $PYTHON setup.py install

RUN rm -rf inference && rm -rf env*
