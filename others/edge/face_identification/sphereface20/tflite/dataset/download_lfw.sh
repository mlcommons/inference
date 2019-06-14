#!/bin/bash


wget http://vis-www.cs.umass.edu/lfw/lfw.tgz -P /tmp/dataset/
wget http://vis-www.cs.umass.edu/lfw/pairs.txt -P /tmp/dataset/
tar zxvf /tmp/dataset/lfw.tgz -C /tmp/dataset

# choose the first 600 pairs
python3 dataset/select_testset.py
