# Copyright (c) 2019, Xilinx, Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import json
import argparse

arg_parser = argparse.ArgumentParser(description="This is a script to convert coco anntations to voc-like annotations.")
arg_parser.add_argument('-ti', '--train_images', type=str, default="./coco2017/train2017", help='where to put coco2017 train images.')
arg_parser.add_argument('-vi', '--val_images', type=str, default='./coco2017/val2017', help='where to put coco2017 val images.')
arg_parser.add_argument('-ta', '--train_anno', type=str, default='./coco2017/instances_train2017.json', help='where to put cooc2017 train set annotations.')
arg_parser.add_argument('-va', '--val_anno', type=str, default='./coco2017/instances_val2017.json', help='where to put coco2017 val set annotations')
arg_parser.add_argument('-tlf', '--tran_list_file', type=str, default='./coco2017/train2017.txt', help='image list for training')
arg_parser.add_argument('-vlf', '--val_list_file', type=str, default='./coco2017/val2017.txt', help='image list for evalution.')
arg_parser.add_argument('-ai', '--all_images', type=str, default='./coco2017/Images', help='where to put all images.')
arg_parser.add_argument('-aa', '--all_anno', type=str, default='./coco2017/Annotations', help='where to put all annotations.')
args = arg_parser.parse_args()

'''How to organize coco dataset folder:
 inputs:
 coco2017/
       |->train2017/
       |->val2017/
       |->instances_train2017.json
       |->instances_val2017.json

outputs:
 coco2017/
       |->Annotations/
       |->Images/
       |->train2017.txt
       |->val2017.txt
'''

def convert_images_coco2voc(args):
    assert os.path.exists(args.train_images)
    assert os.path.exists(args.val_images)
    os.system('mv ' + args.train_images + ' ' + args.all_images)
    os.system('mv ' + args.val_images + '/* ' + args.all_images)
    os.system('rm -r ' + args.val_images)

def generate_cid_name(json_object):
    id2name_dict = {}
    for ind, category_info in enumerate(json_object['categories']):
        id2name_dict[category_info['id']] = category_info['name']
    return id2name_dict

def generate_image_dict(json_object): 
    id2image_dict = {}
    for ind, image_info in enumerate(json_object['images']):
        id2image_dict[image_info['id']] = image_info['file_name']
    return id2image_dict

def generate_annotation_files(json_object, annotation_path, id2image_dict, id2name, image_list_file):
    if not os.path.exists(annotation_path):
        os.mkdir(annotation_path)
    f_image = open(image_list_file, 'w')
    all_images_name = []
    for ind, anno_info in enumerate(json_object['annotations']):
        print('preprocess: {}'.format(ind))
        category_id = anno_info['category_id']
        cls_name = id2name[category_id]       
        image_id = anno_info['image_id']
        image_name = id2image_dict[image_id]
        bbox = anno_info['bbox']
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[3] + bbox[1]
        bbox_str = ' '.join([str(int(x)) for x in bbox])
        with open(os.path.join(annotation_path, image_name.split('.')[0] + '.txt'), 'a') as f_anno:
            f_anno.writelines(image_name.split('.')[0] + " " + cls_name + " " + bbox_str + "\n")
        if image_name not in all_images_name:
            all_images_name.append(image_name)
    for image_name in all_images_name:  
        f_image.writelines(image_name.split('.')[0] + "\n")
    f_image.close() 
                    
def convert_anno_coco2voc(coco_anno_file, image_list_file, all_anno_path):
    with open(coco_anno_file, 'r') as f_ann:
         line = f_ann.readlines()
    json_object = json.loads(line[0])
    id2name = generate_cid_name(json_object)
    id2image_dict = generate_image_dict(json_object)
    generate_annotation_files(json_object, all_anno_path, id2image_dict, id2name, image_list_file)

def convert_anno_all(args):
    convert_anno_coco2voc(args.train_anno, args.tran_list_file, args.all_anno)
    convert_anno_coco2voc(args.val_anno, args.val_list_file, args.all_anno)

if __name__  == "__main__":
    convert_anno_all(args)
    convert_images_coco2voc(args)
