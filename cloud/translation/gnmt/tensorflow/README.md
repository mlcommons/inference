# 1. Problem
- This problem uses recurrent neural network to do language translation.
- The steps to train the model and generate the dataset are listed in [train_gnmt.txt](https://github.intel.com/Intel-MLPerf/inference/blob/master/gnmt/nmt/train_gnmt.txt). Basically, they follow the MLPerf training code. However, you can download the model and dataset with the scripts in this directory.

# 2. Directions

### Install Dependencies

GPU
```
pip install --user tensorflow-gpu
```

CPU
```
pip install --user tensorflow
```

### Run

- go to this folder
```
$ cd /path/to/gnmt/tensorflow/
```

- Change permission and download the pre-trained model and dataset by:

```
$ chmod +x ./download_trained_model.sh
$ ./download_trained_model.sh
$ chmod +x ./download_dataset.sh
$ ./download_dataset.sh
```

- verify the dataset

```
$ chmod +x ./verify_dataset.sh
$ ./verify_dataset.sh
```

- Evaluate performance with a specific batch size.
```
$ python run_task.py --run=performance --batch_size=32
```

- Evaluate accuracy to ensure the target BLEU.
```
$ python run_task.py --run=accuracy
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
BLEU 23.9

---
Questions? Please contact Jerome or Christine at jerome.mitchell@intel.com / christine.cheng@intel.com.



