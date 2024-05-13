#FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt-get update
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    unzip

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY . .
RUN chmod +x ./download_dataset.sh ./verify_dataset.sh  ./download_trained_model.sh ./run.sh

ENTRYPOINT ["python", "mlcube.py"]