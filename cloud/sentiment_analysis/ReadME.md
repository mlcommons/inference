# MLPerf Cloud Inference - Sentiment Analysis Seq2CNN - IMDB dataset

## Dowload the IMDB dataste
To install unzip make sure you have sudo permissions 
'''
  sudo apt-get install unzip
  sh download_imdb.sh
'''
## Dowload the models
Both models use the same set of weights which are given in numpy version as well.
The codes to convert the model from mxnet to pytorch can be found under converting_scripts 
In an environment that has mxnet and numpy run:
'''
    python convert_params_to_numpy.py
'''    
In an environment that has pytorch and numpy run:
'''
    python convert_numpy_to_torch.py
'''    

## Mxnet implemetation
Copy the model using:
'''
  sh downlaod_models_mxnet.sh
'''
## Build docker
'''
  cd mxnet
  docker build -t inference/sentiment.mxnet .
'''
## Run 
docker run -it --rm -v "$(pwd)"/../Datasets:/mlperf/Datasets -v "$(pwd)"/../pretrained_mxnet:/mlperf/pretrained --ipc=host inference/sentiment.mxnet:latest python eval.py --model cnn --eval pretrained/seq2cnn_model --batch-size 1 --calc_accuracy

## Pytorch implementation
Copy the model using:
'''
  sh downlaod_models_pyt.sh
'''
## Build docker
'''
  cd pytorch
  docker build -t inference/sentiment.pytorch .
'''
## Run 
docker run -it --rm -v "$(pwd)"/../Datasets:/mlperf/Datasets -v "$(pwd)"/../pretrained_pyt:/mlperf/pretrained --ipc=host inference/sentiment.pytorch:latest python eval.py  --model cnn --eval pretrained/seq2cnn_imported_mxnet.pth --batch-size 1 

##Model implementation:
### Data and preprocessing
The preprocessing is described in data.py folder and uses spacy tokenizer.
The vocabulary size is 5200
The maximum sequance length is 1000
### Model implementation
The model contain embedding layer, two 1D convolution layers followed by leaky-relu max pooling concatenation and a fully-connected layer the model is described in models/model_cnn.py
Embedding dimension is 1024
CNN1 size [Cin,Cout,kernel]=1024,1024,3
CNN2 size [Cin,Cout,kernel]=1024,1024,4
Fully connected layer 2048->2

Number of parameters:  12,670,978

#To Do
Add SyLT trace generator
Measure results on Intel's NUC

