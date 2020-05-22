# 1. Problem 
Speech recognition accepts raw audio samples and produces a corresponding
character transcription, without an external language model.

# 2. Directions

Open `run.sh`. Set the stage variable to "-1". Set "work_dir" to a
path backed by a disk with at least 2 GB of space. You need conda and
a C/C++ compiler on your PATH. I used conda 4.8.2. This script is
responsible for downloading dependencies, data, and the model.

Run `./run.sh` from this directory.

As you complete stages, you can set the variable "stage" to a higher
number for rerunning.

# 3. Dataset/Environment
### Publication/Attribution
["OpenSLR LibriSpeech Corpus"](http://www.openslr.org/12/) provides over 1000 hours of speech data in the form of raw audio.
We use dev-clean, which is approximately 5 hours. (Note: May subsample if this makes the benchmark too slow.)
### Data preprocessing
What preprocessing is done to the the dataset?

Log filterbanks of size 80 are extracted every 10 milliseconds, from
windows of size 20 milliseconds. Note that every three filterbanks are
concatenated together ("feature splicing"), so the model's effective
frame rate is actually 30 milliseconds.

No dithering takes place.

### Test data order
TODO. Does this even matter for inference?

# 4. Model
This is a variant of the model described in sections 3.1 and 6.2 of:

@article{,
  title={STREAMING END-TO-END SPEECH RECOGNITION FOR MOBILE DEVICES},
  author={Yanzhang He, Tara N. Sainath, Rohit Prabhavalkar, Ian McGraw, Raziel Alvarez, Ding Zhao,
  David Rybach, Anjuli Kannan, Yonghui Wu, Ruoming Pang, Qiao Liang, Deepti Bhatia, Yuan Shangguan,
  Bo Li, Golan Pundak, Khe Chai Sim, Tom Bagby, Shuo-yiin Chang, Kanishka Rao, Alexander Gruenstein},
  journal={arXiv preprint arXiv:1811.06621},
  year={2018}
}

The differences are as follows:

1. The model has 45.3 million parameters, rather than 120 million parameters
1. The LSTMs are not followed by projection layers
1. No layer normalization is used
1. Hidden dimensions are smaller.
1. The prediction network is made of two LSTMs, rather than seven.
1. The labels are characters, rather than word pieces.
1. No quantization is done at this time for inference.
1. A greey decoder is used, rather than a beamsearch decoder. This greatly
   reduces inference complexity.

# 5. Quality
### Quality metric
7.31% Word Error Rate (WER) across all words in the output text of all samples in the
dev-clean set, using a greedy decoder and a fully FP32 model.
