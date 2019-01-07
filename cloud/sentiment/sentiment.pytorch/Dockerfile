FROM pytorch/pytorch

# Set working directory
WORKDIR /mlperf

RUN apt-get update && \
    apt-get install -y python3-tk python3-pip
RUN apt-get install --reinstall build-essential

WORKDIR /mlperf
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m spacy download en
