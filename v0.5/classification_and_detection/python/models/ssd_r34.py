import torch
import torch.nn as nn
from models.base_model_r34 import ResNet34
import numpy as np
from math import sqrt, ceil
import itertools
import torch.nn.functional as F
import torchvision

# From Microsoft/ONNX MLPerf port: https://github.com/BowenBao/inference/tree/master/cloud/single_stage_detector/pytorch
import os
torch.ops.load_library(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../pytorch/lib/', 'custom_ops.cpython-36m-x86_64-linux-gnu.so'))
from torch.onnx import operators

##Inspired by https://github.com/kuangliu/pytorch-ssd


class Encoder(object):
    """
        Transform between (bboxes, lables) <-> SSD output
        
        dboxes: default boxes in size 8732 x 4, 
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format 

        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        #print("# Bounding boxes: {}".format(self.nboxes))
        self.scale_xy = torch.tensor(dboxes.scale_xy)
        self.scale_wh = torch.tensor(dboxes.scale_wh)
    
    
    def decode_batch(self, bboxes_in, scores_in,  criteria = 0.45, max_output=200):
        self.dboxes = self.dboxes.to(bboxes_in)
        self.dboxes_xywh = self.dboxes_xywh.to(bboxes_in)
        bboxes, probs = scale_back_batch(bboxes_in, scores_in,self.scale_xy,self.scale_wh,self.dboxes_xywh)
        boxes = []; labels=[]; scores=[]
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            dbox,dlabel,dscore=self.decode_single(bbox, prob, criteria, max_output)
            boxes.append(dbox)
            labels.append(dlabel)
            scores.append(dscore)
           
        return [boxes,labels,scores]

    # perform non-maximum suppression
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        # Reference to https://github.com/amdegroot/ssd.pytorch
       
        bboxes_out = []        
        scores_out = []
        labels_out = []

        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            if i == 0: continue
            
            score = score.squeeze(1)
            mask = score > 0.05

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0: continue

            score_sorted, score_idx_sorted = score.sort(dim=0)
        
            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []
        
            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < criteria 
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i]*len(candidates))

        bboxes_out, labels_out, scores_out = torch.cat(bboxes_out, dim=0), \
               torch.tensor(labels_out, dtype=torch.long), \
               torch.cat(scores_out, dim=0)


        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]

@torch.jit.script
def calc_iou_tensor(box1, box2):
    """ Calculation of IoU based on two boxes tensor,
        Reference to https://github.com/kuangliu/pytorch-ssd
        input:
            box1 (N, 4) 
            box2 (M, 4)
        output:
            IoU (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    lt = torch.max(be1[:,:,:2], be2[:,:,:2])
    rb = torch.min(be1[:,:,2:], be2[:,:,2:])
    delta = rb - lt
    delta.clone().masked_fill_(delta < 0,0)
    intersect = delta[:,:,0]*delta[:,:,1]
    delta1 = be1[:,:,2:] - be1[:,:,:2]
    area1 = delta1[:,:,0]*delta1[:,:,1]
    delta2 = be2[:,:,2:] - be2[:,:,:2]
    area2 = delta2[:,:,0]*delta2[:,:,1]

    iou = intersect/(area1 + area2 - intersect)
    return iou

@torch.jit.script
def scale_back_batch(bboxes_in, scores_in,scale_xy,scale_wh,dboxes_xywh):
    """
        Do scale and transform from xywh to ltrb
        suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
    """     
    bboxes_in = bboxes_in.permute(0, 2, 1)
    scores_in = scores_in.permute(0, 2, 1)

    bboxes_in[:, :, :2] = scale_xy*bboxes_in[:, :, :2]
    bboxes_in[:, :, 2:] = scale_wh*bboxes_in[:, :, 2:]
    bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*dboxes_xywh[:, :, 2:] + dboxes_xywh[:, :, :2]
    bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp()*dboxes_xywh[:, :, 2:]
    # Transform format to ltrb 
    l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                 bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                 bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                 bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]
    bboxes_in[:, :, 0] = l
    bboxes_in[:, :, 1] = t
    bboxes_in[:, :, 2] = r
    bboxes_in[:, :, 3] = b
    return bboxes_in, F.softmax(scores_in, dim=-1)

@torch.jit.script
def decode_batch_with_nms_trace(bboxes_in:torch.Tensor, scores_in:torch.Tensor, scale_xy:torch.Tensor, scale_wh:torch.Tensor, dboxes_xywh:torch.Tensor): #, criteria:float, max_output:int, device:int=0):
    criteria:float = 0.5
    max_output:int = 200
    device:int = 0

    bboxes, probs = scale_back_batch(bboxes_in, scores_in, scale_xy, scale_wh, dboxes_xywh)

    #assert bboxes.size(0) == 1, 'batch size must be 1'
    bboxes = bboxes.squeeze(0)
    probs = probs.squeeze(0)
    # for each label
    bboxes_out = []
    scores_out = []
    labels_out = []
    # bboxes shape  [box num, 4]
    # probs shape   [box num, label num]
    for i in range(probs.size(1)):
        # skip background
        if i == 0:
            continue

        scores_per_label = probs[:, i]
        mask = scores_per_label > 0.05
        bboxes_masked, scores_masked = bboxes[mask, :], scores_per_label[mask]
        # print('decode single iter scores masked:', scores_masked, scores_masked.shape)

        num_selected = operators.shape_as_tensor(scores_masked)[0].unsqueeze(0)
        k = torch.min(
            torch.cat(
                (torch.tensor([max_output], dtype=torch.long),
                 num_selected), 0))
        _, sorted_idx = scores_masked.topk(k, dim=0)
        bboxes_masked = bboxes_masked[sorted_idx]
        scores_masked = scores_masked[sorted_idx]

        out_idx = torch.ops.roi_ops.nms(bboxes_masked.float(), scores_masked, criteria)

        bboxes_out.append(bboxes_masked[out_idx])
        scores_out.append(scores_masked[out_idx])
        labels_out.append(torch.full_like(out_idx, i, dtype=torch.long))
        # print('decode single iter output:', scores_out[-1], labels_out[-1])
    # return top max_output
    bboxes_out = torch.cat(bboxes_out, dim=0)
    labels_out = torch.cat(labels_out, dim=0)
    scores_out = torch.cat(scores_out, dim=0)

    num_selected = operators.shape_as_tensor(scores_out)[0].unsqueeze(0)
    k = torch.min(
        torch.cat(
            (torch.tensor([max_output], dtype=torch.long), num_selected),
            0
        )
    )
    _, max_ids = scores_out.topk(k, dim=0)

    return bboxes_out[max_ids, :].unsqueeze(0), labels_out[max_ids].unsqueeze(0), scores_out[max_ids].unsqueeze(0)

@torch.jit.script
def decode_batch_with_multi_label_nms_trace(bboxes_in:torch.Tensor, scores_in:torch.Tensor, scale_xy:torch.Tensor, scale_wh:torch.Tensor, dboxes_xywh:torch.Tensor): #, criteria = 0.45, max_output=200, device=0):
    criteria:float = 0.5
    max_output:int = 200
    device:int = 0

    bboxes, probs = scale_back_batch(bboxes_in, scores_in, scale_xy, scale_wh, dboxes_xywh)

    # bboxes shape  [batch, box num, 4]
    # probs shape   [batch, box num, label num]
    probs = probs.permute(0, 2, 1)
    # probs shape   [batch, label num, box num]
    
    # remove background
    probs = probs[:, 1:, :]
    selected_indices = torch.ops.roi_ops.multi_label_nms(bboxes, probs, torch.full((1,), max_output, dtype=torch.long), torch.full((1, ), criteria, dtype=torch.float), torch.full((1, ), 0.05, dtype=torch.float))
    
    labels = selected_indices[:, 1]
    box_indices = selected_indices[:, 2]
    scores_out = probs.reshape(-1)[labels * operators.shape_as_tensor(probs)[2] + box_indices]
    
    # return top max_output
    num_selected = operators.shape_as_tensor(scores_out)[0].unsqueeze(0)
    k = torch.min(
        torch.cat(
            (torch.tensor([max_output], dtype=torch.long), num_selected),
            0
        )
    )
    _, max_ids = scores_out.topk(k, dim=0)
    
    bboxes = bboxes.squeeze(0)[box_indices.index_select(0, max_ids), :].unsqueeze(0)
    labels = labels.index_select(0, max_ids).unsqueeze(0) + 1
    scores_out = scores_out.index_select(0, max_ids).unsqueeze(0)
    
    return bboxes, labels, scores_out

class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, \
                       scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size_w,self.fig_size_h = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh
        
        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps_w = [st[0] for st in steps]
        self.steps_h = [st[1] for st in steps]
        self.scales = scales
        fkw = self.fig_size_w//np.array(self.steps_w)
        fkh = self.fig_size_h//np.array(self.steps_h)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):
            sfeat_w,sfeat_h=sfeat
            sk1 = scales[idx][0]/self.fig_size_w
            sk2 = scales[idx+1][1]/self.fig_size_h
            sk3 = sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]
            for alpha in aspect_ratios[idx]:
                w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat_w), range(sfeat_h)):
                    cx, cy = (j+0.5)/fkh[idx], (i+0.5)/fkw[idx]
                    self.default_boxes.append((cx, cy, w, h)) 
        self.dboxes = torch.tensor(self.default_boxes)
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5*self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5*self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5*self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5*self.dboxes[:, 3]
    
    @property
    def scale_xy(self):
        return self.scale_xy_
    
    @property    
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes

def dboxes_R34_coco(figsize,strides):
    feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]]
    steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
    scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in [21, 45, 99, 153, 207, 261, 315]] 
    aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]] 
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

class SSD_R34(nn.Module):
    """
        Build a SSD module to take 300x300 image input,
        and output 8732 per class bounding boxes

        vggt: pretrained vgg16 (partial) model
        label_num: number of classes (including background 0)
    """
    def __init__(self, label_num=81, backbone='resnet34', model_path="./resnet34-333f7ec4.pth",strides=[3,3 ,2 ,2 ,2 ,2],extract_shapes=False):

        super(SSD_R34, self).__init__()

        self.label_num = label_num
        self.strides = strides
        if backbone == 'resnet34':
            self.model = ResNet34()
            out_channels = 256
            self.out_chan = [out_channels, 512, 512, 256, 256, 256]
        else:
            raise ValueError('Invalid backbone chosen')

        self._build_additional_features(self.out_chan)
        self.extract_shapes=extract_shapes
        # after l2norm, conv7, conv8_2, conv9_2, conv10_2, conv11_2
        # classifer 1, 2, 3, 4, 5 ,6

        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []
        for nd, oc in zip(self.num_defaults, self.out_chan):
            self.loc.append(nn.Conv2d(oc, nd*4, kernel_size=3, padding=1,stride=self.strides[0]))
            self.conf.append(nn.Conv2d(oc, nd*label_num, kernel_size=3, padding=1,stride=self.strides[1]))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        if not extract_shapes:
            self.size=(1200,1200)
            dboxes = dboxes_R34_coco(list(self.size),[3,3,2,2,2,2])
            self.encoder = Encoder(dboxes)
        # intitalize all weights
        self._init_weights()
        self.device = 1

        self.quant = torch.quantization.QuantStub() 
        self.dequant = torch.quantization.DeQuantStub()

    def _build_additional_features(self, input_channels):
        idx = 0
        self.additional_blocks = []
        
        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, input_channels[idx+1], kernel_size=3, padding=1,stride=self.strides[2]),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, input_channels[idx+1], kernel_size=3, padding=1, stride=self.strides[3]),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        # conv9_1, conv9_2
        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, input_channels[idx+1], kernel_size=3, padding=1, stride=self.strides[4]),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        # conv10_1, conv10_2
        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, input_channels[idx+1], kernel_size=3,stride=self.strides[5]),
            nn.ReLU(inplace=True),
        ))
        idx += 1



        # conv11_1, conv11_2
        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, input_channels[idx+1], kernel_size=3),
            nn.ReLU(inplace=True),
        ))

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):

        layers = [
            *self.additional_blocks,
            *self.loc, *self.conf]

        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf,extract_shapes=False):
        ret = []
        features_shapes = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.label_num, -1)))
            # extract shapes for prior box initliziation 
            if extract_shapes:
                ls=l(s)
                features_shapes.append([ls.shape[2],ls.shape[3]])
        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs,features_shapes

    def forward(self, data):
        #data = self.quant(data)

        layers = self.model(data)

        # last result from network goes into additional blocks
        x = layers[-1]
        
        additional_results = []
        for i, l in enumerate(self.additional_blocks):
            
            x = l(x)
            additional_results.append(x)

        src = [*layers, *additional_results]

        locs = []
        confs = []
        for layer_output, loc_conv, conf_conv in zip(src, self.loc, self.conf):
            num_batches:int = 1#layer_output.size(0) 
            num_labels:int = 81#self.label_num
            # Location
            output_loc = loc_conv(layer_output)
            #output_loc = self.dequant(output_loc)
            reshaped_output_loc = output_loc.reshape(num_batches, 4, -1)
            locs.append(reshaped_output_loc)
            # Confidence
            output_conf = conf_conv(layer_output)
            #output_conf = self.dequant(output_conf)
            reshaped_output_conf = output_conf.reshape(num_batches, num_labels, -1)
            confs.append(reshaped_output_conf)
        # Concat all the location tensors, and concat all the confidence tensors:
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
 
        results = decode_batch_with_nms_trace(locs, confs, self.encoder.scale_xy, self.encoder.scale_wh, self.encoder.dboxes_xywh)
        #results = decode_batch_with_multi_label_nms_trace(locs, confs, self.encoder.scale_xy, self.encoder.scale_wh, self.encoder.dboxes_xywh)
        return results

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization.
    # This does not change the numerics but is required by PyTorch.
    def fuse_model(self):
        torch.quantization.fuse_modules(self, ['model.layer1.0', 'model.layer1.1', 'model.layer1.2'], inplace=True)
        for i in range(5):
            torch.quantization.fuse_modules(self, [['additional_blocks.'+str(i)+'.0', 'additional_blocks.'+str(i)+'.1'], ['additional_blocks.'+str(i)+'.2', 'additional_blocks.'+str(i)+'.3']], inplace=True)
        # Fuse ops in R34 the backbone.
        for m in self.modules():
            if type(m) == torchvision.models.quantization.resnet.QuantizableBottleneck:
                m.fuse_model()
            if type(m) == torchvision.models.quantization.resnet.QuantizableBasicBlock:
                m.fuse_model()

