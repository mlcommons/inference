# 1. Problem
- This problem uses recurrent neural network to do language translation.

# 2. Directions

### Install Dependencies

Install python 3.7.2, pytorch 1.0 and cuda 10.
Install other dependencies
```
pip3 install -r requirements.txt
```
This code only has been tested only the versions listed above.

### Run

- Go to the following script which updates translate.py in the MLPerf GNMT training code
```
$ bash ./get_code.sh
```

- Download and verify the dataset.

```
$ cd training/rnn_translator/
$ bash download_dataset.sh
$ bash verify_dataset.sh
```

- Download the pretrained model into the pytorch directory in the training code.

```
$ cd pytorch
$ bash ../../../download_trained_model.sh
```
- Evaluate model accuracy to ensure the model reaches the target accuracy
```
$ python3 translate.py --input ../data/newstest2014.tok.clean.bpe.32000.en \
--output output_file --model model_best.pth \
--reference ../data/newstest2014.de --beam-size 10 \
--math fp32 --dataset-dir ../data --mode accuracy
```

- Evaluate Performance.
```
$ python3 translate.py --input ../data/newstest2014.tok.clean.bpe.32000.en \
--output output_file --model model_best.pth \
--reference ../data/newstest2014.de --beam-size 10 \
--math fp32 --dataset-dir ../data --mode performance
```
There will be an output printout like the following in the performance mode:
```
TEST Time 1.843 (1.843)	Decoder iters 76.0 (76.0)	Tok/s 3818 (3818)
TEST Time 1.597 (2.078)	Decoder iters 57.0 (74.3)	Tok/s 4166 (3588)
TEST Time 1.775 (2.027)	Decoder iters 70.0 (73.4)	Tok/s 3729 (3640)
TEST Time 2.187 (2.009)	Decoder iters 78.0 (74.2)	Tok/s 3744 (3653)
TEST Time 2.038 (2.026)	Decoder iters 79.0 (74.5)	Tok/s 3579 (3639)
TEST SUMMARY:
Lines translated: 3003	Avg total tokens/s: 3648
Avg time per batch: 1.986 s	Avg time per sentence: 33.668 ms
Avg encoder seq len: 28.54	Avg decoder seq len: 28.57	Total decoder iterations: 1780

```

# 3. Dataset

BLEU evaluation is done on newstest2014 from WMT16 English-German
```
@inproceedings{Sennrich2016EdinburghNM,
  title={Edinburgh Neural Machine Translation Systems for WMT 16},
  author={Rico Sennrich and Barry Haddow and Alexandra Birch},
  booktitle={WMT},
  year={2016}
}
```

# 4. Model

This code is modified from github: https://github.com/tensorflow/nmt

```
@article{wu2016google,
  title={Google's neural machine translation system: Bridging the gap between human and machine translation},
  author={Wu, Yonghui and Schuster, Mike and Chen, Zhifeng and Le, Quoc V and Norouzi, Mohammad and Macherey, Wolfgang and Krikun, Maxim and Cao, Yuan and Gao, Qin and Macherey, Klaus and others},
  journal={arXiv preprint arXiv:1609.08144},
  year={2016}
}

```
# 5. Quality.
### Quality metric
BLEU 22.16

---
Questions? Please contact christine.cheng@intel.com.
