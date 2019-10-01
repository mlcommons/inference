# 1. Problem
- This problem uses recurrent neural network to do language translation.
- The steps to train the model and generate the dataset are listed in [train_gnmt.txt](https://github.com/mlperf/inference/blob/master/v0.5/translation/gnmt/tensorflow/train_gnmt.txt). Basically, they follow the MLPerf training code. However, you can download the model and dataset with the scripts in this directory.

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

### Run GNMT over full Dataset

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

### Run GNMT through LoadGen:
1.  For LoadGen introduction, please refer to https://github.com/mlperf/inference/blob/master/loadgen/README.md
Follow the instructions to install LoadGen from https://github.com/mlperf/inference/blob/master/loadgen/README_BUILD.md

2.  Run:
```
python loadgen_gnmt.py --store_translation
```

This will invoke the SingleStream scenario (default --scenario option) in Performance mode (default --mode option), and in addition, will store the output of every sentence in a separate file.

Other scenarios can be ran by changing the "--scenario" option. Accuracy tracking can be enabled with the "--mode Accuracy" option. Debugging settings can be enabled with "--debug_settings". Please run the following command for complete overview of options:
```
python loadgen_gnmt.py -h
```

To check accuracy, please run the following commands:
```
python loadgen_gnmt.py --mode Accuracy
python process_accuracy.py
```

Please ensure the performance mode uses nmt/data/newstest2014.tok.bpe.32000.en.large and accuracy mode uses nmt/data/newstest2014.tok.bpe.32000.en from the dataset link. 


### Running other datasets:
In order to translate other English texts, the raw text needs to be preprocessed first:
1. Ensure you have an English text, along with it's German translation, suffixed with, ".en" and ".de", respectively (e.g., newstest2014.en and newstest2014.de).
2. Run the following command:
```
./preprocess_input.sh newstest2014 
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



