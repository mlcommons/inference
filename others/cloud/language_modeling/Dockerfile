FROM ubuntu:18.04
  
# Set working directory
WORKDIR /mlperf

RUN apt-get update && \
    apt-get install -y python3 python3-pip

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8 

# Necessary pip packages
COPY requirements.txt /mlperf/
RUN pip3 install -r /mlperf/requirements.txt

WORKDIR /mlperf
