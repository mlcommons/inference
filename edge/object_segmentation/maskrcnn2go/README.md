# MaskRCNN2GO MLPerf submission

Owner: Peizhao Zhang (stzpz@fb.com)

Model: MaskRCNN2GO (bbox + segmentation) float32

Datasets used for evaluation: COCO 2014 minival

Input: 
  * data (1, 3, H, W), min(H, W) = 320, BGR in range [0, 255]
  * im_info (1, 3) [scaled_height, scaled_width, scale]

Proposals (pre nms, post nms): 3000/100

Accuracy(mAP[IoU=0.50:0.95]): 
	float32: 	25.1 (bbox), 21.6 (segmentation)

Evaluation:
* Download COCO 2014 minival dataset
* Update path in run_eval.sh
* Run run_eval.sh


## Acknowledgement

Thanks a lot for the help from Carole-Jean Wu and Fei Sun.

