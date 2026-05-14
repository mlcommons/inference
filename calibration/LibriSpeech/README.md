# Calibration set for RNN-T benchmark

These are 500 sequences randomly selected from the **train-clean-100** set in [LibriSpeech](http://www.openslr.org/12/) dataset and to be used as the reference calibration set for MLPerf Inference RNN-T benchmark. 

## Criteria

1. The calibration set and the test set should not have the same speakers.
2. The calibration set and the test set should not have the same sentences as labels.
3. The model should get about the same WER on the calibration set and the test set.
4. The calibration data should represent all characters in the alphabet. The distribution of characters does not need to be uniform or an unbiased estimate of the full dataset's distribution, though.
5. The sequences in the calibration set are all below 15 seconds long.

