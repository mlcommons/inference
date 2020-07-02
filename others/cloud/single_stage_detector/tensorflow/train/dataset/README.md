## Prepare Dataset
```
0. dowload dataset:
   bash download_dataset.sh 
1. reorg coco2017 dataset as convert_coco2voc_like.py description.
2. convert coco datset to voc-like dataset.
   python convert_coco2voc_like.py
3. generate tfrecords.
   python convert_tfrecords.py
```
