# About
This version of DeepSpeech2 is adapted from MLPerf for the purpose of providing you a useable model for speech recognition.
You can find the MLPerf repository: https://github.com/mlperf/training/commit/9c6b8b3ed27c2c8bc1e9f697b9d7c0ed75a72efb
You can find our pytorch and ONNX models: https://drive.google.com/drive/u/1/folders/1K3aIu2qm1R2h55C4-qgGlYOOID6sohru

# Directions
The following outlines the workflow for those who intend to run both training and inference.

### Workflow
1. Environment set up
2. Download & preprocess dataset
3. Run training
4. Run inference

Please see the corresponding folders for detailed instructions.

# Model
### Publication/Attribution
This is an implementation of [DeepSpeech2](https://arxiv.org/pdf/1512.02595.pdf) adapted from [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch).
### List of layers
Summary: Sampled Audio Spectrograms -> 2 CNN layers -> 5 Bi-Directional GRU layers -> FC classifier layer -> output text

Details:

  (module): DeepSpeech (

    (conv): Sequential (

      (0): Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2))

      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)

      (2): Hardtanh (min_val=0, max_val=20, inplace)

      (3): Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1))

      (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)

      (5): Hardtanh (min_val=0, max_val=20, inplace)

    )
    (rnns): Sequential (

      (0): BatchRNN (


        (rnn): GRU(672, 2560)
      )

      (1): BatchRNN (

        (rnn): GRU(2560, 2560)

      )

      (2): BatchRNN (

        (rnn): GRU(2560, 2560)

      )

    )

    (fc): Sequential (

      (0): SequenceWise (

      Sequential (

        (0): BatchNorm1d(2560, eps=1e-05, momentum=0.1, affine=True)

        (1): Linear (2560 -> 29)

      ))

    )

    (inference_log_softmax): InferenceBatchLogSoftmax (

    )

  )

)

### Model Weights

- Stored on https://drive.google.com/drive/u/1/folders/1OioL2tqOsVWNW0j_I6J7gZxneFBd2gsB

- There are 4 files, 2 onnx conversion and 2 pytorch weights. The blue icons are onnx conversions, the rest are pytorch model weights


# Key Result Summary

A result summary can be found via google doc https://docs.google.com/document/d/1sDra8B-g7GXLaza33iCUlDEK6_5KZQ23bwszyJUPSYA/edit?usp=sharing

- Inference results
    - results under https://github.com/tbd-ai/tbd-suite/tree/master/SpeechRecognition-DeepSpeech2/pytorch/results/inference

- Training results
    - results under https://github.com/tbd-ai/tbd-suite/tree/master/SpeechRecognition-DeepSpeech2/pytorch/results/training

### Quality

|Epoch		|Test		|Validation			|Hyperparameter Notes|
|-----------|-----------|-------------------|--------------------|
|Start		|--			|82.4				|MLPerf Default|
|3			|--			|41.9				|MLPerf Default|
|6			|--			|37.5				|MLPerf Default|
|9			|--			|35.4				|Original Paper|
|13			|--			|34.1				|"Aggressive LR"|
|20			|21.1		|33.7				|"Aggressive LR"|

Please see the results summary doc for details on the hyperparameters: https://docs.google.com/document/d/1sDra8B-g7GXLaza33iCUlDEK6_5KZQ23bwszyJUPSYA/edit?usp=sharing

### Performance

You can see performance plots here: https://docs.google.com/document/d/1sDra8B-g7GXLaza33iCUlDEK6_5KZQ23bwszyJUPSYA/edit?usp=sharing

- Roughly 3x realtime speed up (audio clip duration / latency) on CPU

- 99% of inputs with a distribution comparable to Librispeech Test Clean completes in 9.17s (on our local machine) and 10.07 (on Azure F8s_v2)

- 20 seconds of audio choped into 2s clips and batched together has 99%-tile latency of roughtly 1s

- See https://github.com/tbd-ai/tbd-suite/tree/master/SpeechRecognition-DeepSpeech2/pytorch/results/inference for more graphs and details
