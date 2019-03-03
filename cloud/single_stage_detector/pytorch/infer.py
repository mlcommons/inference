import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd_r34 import SSD_R34
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np


def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='../coco',
                        help='path to test and training data files')
    parser.add_argument('--batch-size', '-b', type=int, default=4,
                        help='number of examples for each iteration')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--device', '-did', type=int,
                        help='device id')                    
    parser.add_argument('--threshold', '-t', type=float, default=0.212,
                        help='stop training early at threshold')
    parser.add_argument('--checkpoint', type=str, default='./pretrained/resnet34-ssd1200.pth',
                        help='path to model checkpoint file')
    parser.add_argument('--image-size', default=[1200,1200], type=int, nargs='+',
                        help='input image sizes (e.g 1400 1400,1200 1200')  
    parser.add_argument('--strides', default=[3,3,2,2,2,2], type=int, nargs='+',
                        help='stides for ssd model must include 6 numbers')                                       
    parser.add_argument('--use-fp16', action='store_true')                          
    return parser.parse_args()


def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


def dboxes_R34_coco(figsize,strides):
    ssd_r34=SSD_R34(81,strides=strides)
    synt_img=torch.rand([1,3]+figsize)
    #if use_cude:
    #    synt_img.to('cuda')
    #    ssd_r34.to('cuda')
    _,_,feat_size =ssd_r34(synt_img, extract_shapes = True)
    print('Features size: ', feat_size)
    import pdb; pdb.set_trace()
    steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in [21, 45, 99, 153, 207, 261, 315]] 
    aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]] 
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes
    
def coco_eval(model, coco, cocoGt, encoder, inv_map, threshold,device=0,use_cuda=False):
    from pycocotools.cocoeval import COCOeval
    model.eval()
    if use_cuda:
        model = model.to('cuda')
    ret = []
    start = time.time()
    for idx, image_id in enumerate(coco.img_keys):
        img, (htot, wtot), _, _ = coco[idx]

        with torch.no_grad():
            print("Parsing image: {}/{}".format(idx+1, len(coco)), end="\r")
            inp = img.unsqueeze(0)
            if use_cuda:
                inp = inp.to('cuda')
            start_time=time.time()
            ploc, plabel,_ = model(inp)
            time.time()-start_time
            print('Mode inference time: ', time.time()-start_time)
            try:
                result = encoder.decode_batch(ploc, plabel, 0.50, 200,device=device)[0]
            except:
                #raise
                print("No object detected in idx: {}".format(idx))
                continue
            print('Decoding time: ', time.time()-start_time)
            loc, label, prob = [r.cpu().numpy() for r in result]
            
            for loc_, label_, prob_ in zip(loc, label, prob):
                ret.append([image_id, loc_[0]*wtot, \
                                      loc_[1]*htot,
                                      (loc_[2] - loc_[0])*wtot,
                                      (loc_[3] - loc_[1])*htot,
                                      prob_,
                                      inv_map[label_]])
    print("")
    print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))
    cocoDt = cocoGt.loadRes(np.array(ret))

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))

    return (E.stats[0] >= threshold) #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]



def eval_ssd_r34_mlperf_coco(args):
    from coco import COCO
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    dboxes = dboxes_R34_coco(args.image_size,args.strides)
    encoder = Encoder(dboxes)
    val_trans = SSDTransformer(dboxes, (args.image_size[0], args.image_size[1]), val=True)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")

    cocoGt = COCO(annotation_file=val_annotate)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    inv_map = {v:k for k,v in val_coco.label_map.items()}

    ssd_r34 = SSD_R34(val_coco.labelnum,args.strides)

    print("loading model checkpoint", args.checkpoint)
    od = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    import pdb; pdb.set_trace()
    ssd_r34.load_state_dict(od["model"])

    if use_cuda:
        ssd_r34.cuda(args.device)
    loss_func = Loss(dboxes)
    if use_cuda:
        loss_func.cuda(args.device)

    coco_eval(ssd_r34, val_coco, cocoGt, encoder, inv_map, args.threshold,args.device,use_cuda)

def main():
    args = parse_args()

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark = True
    eval_ssd_r34_mlperf_coco(args)

if __name__ == "__main__":
    main()
