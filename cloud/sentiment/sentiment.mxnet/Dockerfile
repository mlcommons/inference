FROM mxnet/python

# Set working directory
WORKDIR /mlperf

RUN apt-get update && \
    apt-get install -y python3-tk python3-pip
RUN apt-get install --reinstall build-essential

WORKDIR /mlperf
COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en
