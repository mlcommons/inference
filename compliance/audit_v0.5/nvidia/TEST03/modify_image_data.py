#! /usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd())

import argparse
import numpy as np
import shutil

#from common import logging
#from PIL import Image
import cv2
import math


def modify_imagenet(data_dir, custom_data_dir):

    #logging.info("Modifying imagenet...")
    print("Moidfying imagenet")
    dirlist = os.listdir(data_dir)
    image_list = [x for x in dirlist if x.endswith(".JPEG")]

    src_dir = data_dir
    dst_dir = os.path.join(custom_data_dir, "imagenet")

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
		
    for idx, file_name in enumerate(image_list):
        if (idx % 1000) == 0: 
            print("Processing image No.{:d}/{:d}...".format(idx, len(image_list)))
        img_out = os.path.join(dst_dir, file_name)
        if not os.path.exists(img_out):
            image = cv2.imread(os.path.join(src_dir, file_name))
            #Set pixels to 0
            image[:,:,0] = 0
            #print ("Writing image No.{:d}/{:d}...".format(idx, len(image_list)))
            cv2.imwrite(img_out, image)


def modify_coco(data_dir, custom_data_dir):

    #logging.info("Preprocessing coco...")

    def modify_coco_helper(src_dir, dst_dir, image_list):

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for idx, file_name in enumerate(image_list):
            #logging.info("Processing image No.{:d}/{:d}...".format(idx, len(image_list)))
            img_out = os.path.join(dst_dir, file_name)
            if not os.path.exists(img_out):
                image_path = os.path.join(src_dir, file_name)
                image = cv2.imread(image_path)
                #Set pixels to 0
                image[:,:,0] = 0
                cv2.imwrite(img_out, image)

    #Modify the validation set
    src_dir = os.path.join(data_dir, "val2017")
    dst_dir = os.path.join(custom_data_dir, "coco/val2017/")

    dirlist = os.listdir(src_dir)
    image_list = [x for x in dirlist if x.endswith(".jpg")]
    modify_coco_helper(src_dir, dst_dir, image_list)

    #Copy the training set
    src_dir = os.path.join(data_dir, "train2017")
    dst_dir = os.path.join(custom_data_dir, "coco/train2017")
    shutil.copytree(src_dir, dst_dir)

def copy_coco_annotations(data_dir, output_dir):
    src_dir = os.path.join(data_dir, "annotations")
    dst_dir = os.path.join(output_dir, "coco/annotations")
    shutil.copytree(src_dir, dst_dir)

def main():
    # Parse arguments to identify the data directory with the input images
    #   and the output directory for the new custom images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-d",
        help="Specifies the directory containing the input images.",
        default=""
    )
    parser.add_argument(
        "--output_dir", "-o",
        help="Specifies the output directory for the custom data.",
        default=""
    )
    parser.add_argument(
        "--dataset",
        help="Specifies the dataset - coco or imagenet",
        default=""
    )
    args = parser.parse_args()
    print ("Running dataset modifer....")
    # Now, actually modify the input images
    #logging.info("Loading and modifying images. This might take a while...")
    data_dir = args.data_dir
    output_dir = args.output_dir
    #while True:
        #print ("a")
        #pass
    if args.dataset == "imagenet":
        print("Begin Imagenet")
        modify_imagenet(data_dir, output_dir)
        print("Imagenet complete")
    elif args.dataset == "coco":
        modify_coco(data_dir, output_dir)
        copy_coco_annotations(data_dir, output_dir)
    else:
        print("Incorrect dataset")
        #logging.info("Incorrect dataset. It can be either coco or imagenet.")
    #logging.info("Processing done.")

if __name__ == '__main__':
	main()

