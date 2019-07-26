# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

import argparse
import os, sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from IPython import embed

def cocoval(detected_json, eval_json):
    eval_gt = COCO(eval_json)
    eval_dt = eval_gt.loadRes(detected_json)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str, help='json to eval')
    args = parser.parse_args()
