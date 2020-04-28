# 1. Problem 
Speech recognition accepts raw audio samples and produces a corresponding
character transcription, without an external language model.

# 2. Directions

Open `steps/run.sh`. Set the stage variable to "-1". You need conda on
your PATH.

Run `steps/run.sh`.

As you complete stages, you can set the variable "stage" to a higher
number for rerunning.

NOTE: This part will be elaborated on later. There is no Loadgen
integration right now.

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

TODO: Figure out if preprocessing is considered part of the
benchmark. For streaming inference, it definitely should be part of
the benchmark. For training, it should not be part of the
benchmark. For offline inference, it probably shouldn't be part of the
benchmark, but I'm not sure. It depends upon whether your offline
workload will be running more the one model on the inputs (e.g., voice
activity detection, followed by actual speech-to-text. I am fairly
skeptical that computing log mel spectra is particularly compute
intensive.) Slide 11 here confirms that skipping featurization can be
reasonable for offline use case:
https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9672-accelerate-your-speech-recognition-pipeline-on-the-gpu.pdf It's not a bottleneck.

### Test data order
In what order is the test data traversed? TODO

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
1. No layernormalization is used
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
