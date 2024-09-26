FROM ubuntu:18.04

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
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash ./miniconda.sh -b -p /opt/anaconda3 && \
    rm miniconda.sh && \
    /opt/anaconda3/bin/conda clean -tipsy && \
    ln -s /opt/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    conda config --set always_yes yes --set changeps1 no

RUN conda install pytorch torchvision cpuonly -c pytorch
RUN pip install fbgemm-gpu==0.3.2 tqdm torchmetrics==0.11.0 torchrec==0.3.2 torchsnapshot pyre-extensions scikit-learn --user
RUN pip install fbgemm-gpu-cpu==0.3.2
RUN pip install future pillow onnx opencv-python-headless tensorflow==2.4 onnxruntime
RUN pip install tensorboard
RUN pip install Cython && pip install pycocotools
RUN pip install scikit-learn
RUN pip install numpy
RUN pip install pydot
RUN pip install torchviz
RUN pip install protobuf
RUN pip install tqdm
RUN pip install pybind11

RUN cd /tmp && \
    git clone --recurse-submodules https://github.com/mlcommons/inference && \
    cd inference/loadgen && \
    CFLAGS="-std=c++14" python setup.py install && \
    cd ../../ && \
    rm -rf inference


ENTRYPOINT ["/bin/bash"]
