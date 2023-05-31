FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

ENV PYTHON_VERSION=3.8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PATH /opt/anaconda3/bin:$PATH

WORKDIR /root
ENV HOME /root

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
    unzip

RUN cd /opt && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash ./miniconda.sh -b -p /opt/anaconda3 && \
    rm miniconda.sh && \
    /opt/anaconda3/bin/conda clean -tipsy && \
    ln -s /opt/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    conda config --set always_yes yes --set changeps1 no

RUN pip install --upgrade pip && \
    pip install cython future pillow onnx opencv-python-headless && \
    ln -s /usr/local/cuda/lib64 /usr/local/cuda/lib && \
    cp /usr/lib/x86_64-linux-gnu/libnccl* /usr/local/cuda/lib && \
    ldconfig

RUN conda install pytorch torchvision -c pytorch
RUN pip install fbgemm-gpu==0.3.2 tqdm torchmetrics==0.11.0 torchrec==0.3.2 torchsnapshot pyre-extensions scikit-learn --user
RUN pip install tensorflow-gpu==2.4 onnxruntime-gpu
RUN pip install tensorboard
RUN pip install Cython && pip install pycocotools
RUN pip install -U scikit-learn


RUN cd /tmp && \
    git clone --recurse-submodules https://github.com/mlcommons/inference && \
    cd inference/loadgen && \
    CFLAGS="-std=c++14" python setup.py install && \
    cd ../../ && \
    rm -rf inference

RUN  echo "/usr/local/cuda/compat" >> /etc/ld.so.conf.d/cuda-10-0.conf && \
    ldconfig

ENTRYPOINT ["/bin/bash"]
