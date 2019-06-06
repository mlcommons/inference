"""
implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import json
import logging
import os
import time

import numpy as np
from PIL import Image
from pycocotools.cocoeval import COCOeval
import pycoco
import dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


class Coco(dataset.Dataset):
    def __init__(self, data_path, image_list, name, use_cache=0, image_size=None,
                 image_format="NHWC", pre_process=None, count=None, cache_dir=None):
        super().__init__()
        self.image_size = image_size
        self.image_list = []
        self.label_list = []
        self.image_ids = []
        self.image_sizes = []
        self.count = count
        self.use_cache = use_cache
        self.data_path = data_path
        self.pre_process = pre_process
        if not cache_dir:
            cache_dir = os.getcwd()
        self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format)
        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False
        not_found = 0
        if image_list is None:
            # by default look for val_map.txt
            image_list = os.path.join(data_path, "annotations/instances_val2017.json")
        self.annotation_file = image_list

        os.makedirs(self.cache_dir, exist_ok=True)
        start = time.time()
        images = {}
        with open(image_list, "r") as f:
            coco = json.load(f)
        for i in coco["images"]:
            images[i["id"]] = {"file_name": i["file_name"],
                               "height": i["height"],
                               "width": i["width"],
                               "bbox": [],
                               "category": []}
        for a in coco["annotations"]:
            i = images.get(a["image_id"])
            if i is None:
                continue
            i["category"].append(a.get("category_id"))
            i["bbox"].append(a.get("bbox"))

        for image_id, img in images.items():
            image_name = os.path.join("val2017", img["file_name"])
            src = os.path.join(data_path, image_name)
            if not os.path.exists(src):
                # if the image does not exists ignore it
                not_found += 1
                continue
            os.makedirs(os.path.dirname(os.path.join(self.cache_dir, image_name)), exist_ok=True)
            dst = os.path.join(self.cache_dir, image_name)
            if not os.path.exists(dst + ".npy"):
                # cache a preprocessed version of the image
                with Image.open(src) as img_org:
                    processed = self.pre_process(img_org, need_transpose=self.need_transpose, dims=self.image_size)
                    np.save(dst, processed)

            self.image_ids.append(image_id)
            self.image_list.append(image_name)
            self.image_sizes.append((img["height"], img["width"]))
            self.label_list.append((img["category"], img["bbox"]))

            # limit the dataset if requested
            if self.count and len(self.image_list) > self.count:
                break

        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if not_found > 0:
            log.info("reduced image list, %d images not found", not_found)

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, time_taken))

        self.label_list = np.array(self.label_list)

    def get_item(self, nr):
        """Get image by number in the list."""
        dst = os.path.join(self.cache_dir, self.image_list[nr])
        img = np.load(dst + ".npy")
        return img, self.label_list[nr]

    def get_item_loc(self, nr):
        src = os.path.join(self.data_path, self.image_list[nr])
        return src


class PostProcessCoco:
    """
    Post processing for tensorflow ssd-mobilenet style models
    """
    def __init__(self):
        self.results = []
        self.good = 0
        self.total = 0
        self.use_inv_map = False

    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, ids, expected=None, result_dict=None, ):
        # results come as:
        #   tensorflow, ssd-mobilenet: num_detections,detection_boxes,detection_scores,detection_classes
        processed_results = []
        # batch size
        bs = len(results[0])
        for idx in range(0, bs):
            processed_results.append([])
            detection_num = int(results[0][idx])
            detection_boxes = results[1][idx]
            detection_classes = results[3][idx]
            expected_classes = expected[idx][0]
            for detection in range(0, detection_num):
                detection_class = int(detection_classes[detection])
                if detection_class in expected_classes:
                    self.good += 1
                box = detection_boxes[detection]
                processed_results[idx].append([float(ids[idx]),
                                              box[0], box[1], box[2], box[3],
                                              results[2][idx][detection],
                                              float(detection_class)])
                self.total += 1
        return processed_results

    def start(self):
        self.results = []
        self.good = 0
        self.total = 0

    def finalize(self, result_dict, ds=None, output_dir=None):
        result_dict["good"] += self.good
        result_dict["total"] += self.total
        image_ids = []

        if self.use_inv_map:
            # for pytorch
            label_map = {}
            with open(ds.annotation_file) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1
            inv_map = {v:k for k,v in label_map.items()}

        detections = []
        for batch in range(0, len(self.results)):
            for idx in range(0, len(self.results[batch])):
                detection = self.results[batch][idx]
                # this is the index into the image list
                #image_id = int(detections[idx][0])
                image_id = int(detection[0])
                image_ids.append(image_id)
                # map it to the coco image it
                detection[0] = ds.image_ids[image_id]
                height, width = ds.image_sizes[image_id]
                # box comes from model as: ymin, xmin, ymax, xmax
                ymin = detection[1] * height
                xmin = detection[2] * width
                ymax = detection[3] * height
                xmax = detection[4] * width
                # pycoco wants {imageID,x1,y1,w,h,score,class}
                detection[1] = xmin
                detection[2] = ymin
                detection[3] = xmax - xmin
                detection[4] = ymax - ymin
                if self.use_inv_map:
                    cat_id = inv_map.get(int(detection[6]), -1)
                    if cat_id == -1:
                        # FIXME:
                        log.info("finalize can't map category {}".format(int(detection[6])))
                    detection[6] =  cat_id
                detections.append(np.array(detection))

        # for debugging
        if output_dir:
            # for debugging
            pp = []
            for detection in detections:
                pp.append({"image_id": int(detection[0]),
                           "image_loc": ds.get_item_loc(image_ids[idx]),
                           "category_id": int(detection[6]),
                           "bbox": [float(detection[1]), float(detection[2]),
                                    float(detection[3]), float(detection[4])],
                           "score": float(detection[5])})
            if not output_dir:
                output_dir = "/tmp"
            fname = "{}/{}.json".format(output_dir, result_dict["scenario"])
            with open(fname, "w") as fp:
                json.dump(pp, fp, sort_keys=True, indent=4)


        image_ids = list(set([i[0] for i in detections]))
        self.results = []
        cocoGt = pycoco.COCO(ds.annotation_file)
        cocoDt = cocoGt.loadRes(np.array(detections))
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.params.imgIds = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        result_dict["mAP"] = cocoEval.stats[0]


class PostProcessCocoPt(PostProcessCoco):
    """
    Post processing required by ssd-resnet34 / pytorch
    """
    def __init__(self,use_inv_map,score_threshold):
        super().__init__()
        self.use_inv_map = use_inv_map
        self.score_threshold = score_threshold
        
    def __call__(self, results, ids, expected=None, result_dict=None):
        # results come as:
        #   detection_boxes,detection_classes,detection_scores

        processed_results = []
        # batch size
        bs = len(results[0])
        for idx in range(0, bs):
            #processed_results.append([])
            detection_boxes = results[0][idx]
            detection_classes = results[1][idx]
            expected_classes = expected[idx][0]
            scores = results[2][idx]
            #for detection in range(0, len(expected_classes)):
            for detection in range(0, len(scores)):
                if scores[detection] < self.score_threshold:
                    break
                detection_class = int(detection_classes[detection])
                if detection_class in expected_classes:
                    self.good += 1
                box = detection_boxes[detection]
                # comes from model as:  0=xmax 1=ymax 2=xmin 3=ymin
                processed_results.append([[float(ids[idx]),
                                              box[1], box[0], box[3], box[2],
                                              scores[detection],
                                              float(detection_class)]])
                self.total += 1
        return processed_results


class PostProcessCocoOnnx(PostProcessCoco):
    """
    Post processing required by ssd-resnet34 / onnx
    """
    def __init__(self):
        super().__init__()

    def __call__(self, results, ids, expected=None, result_dict=None):
        # results come as:
        #   onnx (from pytorch ssd-resnet34): detection_boxes,detection_classes,detection_scores

        processed_results = []

        # batch size
        bs = len(results[0])
        for idx in range(0, bs):
            processed_results.append([])
            detection_boxes = results[0][idx]
            detection_classes = results[1][idx]
            expected_classes = expected[idx][0]
            scores = results[2][idx]
            for detection in range(0, len(scores)):
                if scores[detection] < 0.5:
                    break
                detection_class = int(detection_classes[detection])
                if detection_class in expected_classes:
                    self.good += 1
                box = detection_boxes[detection]
                # comes from model as:  0=xmax 1=ymax 2=xmin 3=ymin
                processed_results.append([float(ids[idx]),
                                              box[1], box[0], box[3], box[2],
                                              scores[detection],
                                              float(detection_class)])
                self.total += 1
        return results

class PostProcessCocoTf(PostProcessCoco):
    """
    Post processing required by ssd-resnet34-tf
    """
    def __init__(self):
        super().__init__()
        self.use_inv_map = True
        figsize = [1200, 1200]
        strides = [3, 3, 2, 2, 2, 2]
        feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]]
        steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
        # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
        scales = [(int(s*figsize[0] / 300),int(s * figsize[1] / 300)) for s in [21, 45, 99, 153, 207, 261, 315]]
        aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        dboxes = self.DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
        self.encoder = self.Encoder(dboxes)

    class DefaultBoxes(object):
        def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, \
                           scale_xy=0.1, scale_wh=0.2):
            from math import sqrt
            from itertools import product
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
                    for i, j in product(range(sfeat_w), range(sfeat_h)):
                        cx, cy = (j+0.5)/fkh[idx], (i+0.5)/fkw[idx]
                        self.default_boxes.append((cx, cy, w, h))
            self.dboxes = np.array(self.default_boxes)
            self.dboxes.clip(min=0, max=1, out=self.dboxes)
            # For IoU calculation
            self.dboxes_ltrb = self.dboxes.copy()
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

    class Encoder(object):
        def __init__(self, dboxes):
            self.dboxes = dboxes(order="ltrb")
            self.dboxes_xywh = np.expand_dims(dboxes(order="xywh"),0)
            self.nboxes = self.dboxes.shape[0]
            self.scale_xy = dboxes.scale_xy
            self.scale_wh = dboxes.scale_wh

        @staticmethod
        def softmax_cpu(x, dim=-1):
            x = np.exp(x)
            s = np.expand_dims(np.sum(x, axis=dim), dim)
            return x/s

        # This function is from https://github.com/kuangliu/pytorch-ssd.
        @staticmethod
        def calc_iou_tensor(box1, box2):
            """ Calculation of IoU based on two boxes tensor,
                Reference to https://github.com/kuangliu/pytorch-ssd
                input:
                    box1 (N, 4)
                    box2 (M, 4)
                output:
                    IoU (N, M)
            """
            N = box1.shape[0]
            M = box2.shape[0]

            be1 = np.expand_dims(box1, 1).repeat(M, axis=1)
            be2 = np.expand_dims(box2, 0).repeat(N, axis=0)
            lt = np.maximum(be1[:,:,:2], be2[:,:,:2])
            rb = np.minimum(be1[:,:,2:], be2[:,:,2:])

            delta = rb - lt
            delta[delta < 0] = 0
            intersect = delta[:,:,0]*delta[:,:,1]

            delta1 = be1[:,:,2:] - be1[:,:,:2]
            area1 = delta1[:,:,0]*delta1[:,:,1]
            delta2 = be2[:,:,2:] - be2[:,:,:2]
            area2 = delta2[:,:,0]*delta2[:,:,1]

            iou = intersect/(area1 + area2 - intersect)
            return iou

        def scale_back_batch(self, bboxes_in, scores_in):
            """
                Do scale and transform from xywh to ltrb
                suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
            """

            bboxes_in = bboxes_in.transpose([0,2,1])
            scores_in = scores_in.transpose([0,2,1])

            bboxes_in[:, :, :2] = self.scale_xy*bboxes_in[:, :, :2]
            bboxes_in[:, :, 2:] = self.scale_wh*bboxes_in[:, :, 2:]

            bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
            bboxes_in[:, :, 2:] = np.exp(bboxes_in[:, :, 2:])*self.dboxes_xywh[:, :, 2:]

            # Transform format to ltrb
            l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                         bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                         bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                         bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

            bboxes_in[:, :, 0] = l
            bboxes_in[:, :, 1] = t
            bboxes_in[:, :, 2] = r
            bboxes_in[:, :, 3] = b

            return bboxes_in, self.softmax_cpu(scores_in, dim=-1)

        def decode_batch(self, bboxes_in, scores_in,  criteria = 0.45, max_output=200):
            bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)
            output = []
            #for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            #    bbox = bbox.squeeze(0)
            #    prob = prob.squeeze(0)
            for bbox, prob in zip(bboxes, probs):
                output.append(self.decode_single(bbox, prob, criteria, max_output))
                #print(output[-1])
            return output

        # perform non-maximum suppression
        def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
            # Reference to https://github.com/amdegroot/ssd.pytorch

            bboxes_out = []
            scores_out = []
            labels_out = []

            for i, score in enumerate(np.split(scores_in, scores_in.shape[1], 1)):
                if i == 0: continue
                score = score.squeeze(1)
                mask = score > 0.05

                bboxes, score = bboxes_in[mask, :], score[mask]
                if score.shape[0] == 0: continue
                score_idx_sorted = np.argsort(-score, axis=0)

                # select max_output indices
                #score_idx_sorted = score_idx_sorted[-max_num:]
                score_idx_sorted = score_idx_sorted[:max_num]
                candidates = []
                while score_idx_sorted.size > 0:
                    idx = score_idx_sorted[0].item()
                    bboxes_sorted = bboxes[score_idx_sorted, :]
                    bboxes_idx = np.expand_dims(bboxes[idx, :],0)
                    iou_sorted = self.calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()
                    # we only need iou < criteria
                    score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                    candidates.append(idx)

                bboxes_out.append(bboxes[candidates, :])
                scores_out.append(score[candidates])
                labels_out.extend([i]*len(candidates))

            bboxes_out = np.concatenate(bboxes_out, axis=0)
            labels_out = np.array(labels_out, dtype=np.long)
            scores_out = np.concatenate(scores_out, axis=0)
            max_ids = np.argsort(-scores_out, axis=0)
            max_ids = max_ids[:max_output]
            return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


    def __call__(self, results, ids, expected=None, result_dict=None):
        # results come as:
        #   onnx (from pytorch ssd-resnet34): detection_boxes,detection_classes,detection_scores

        processed_results = []
        #loc, label, prob = self.encoder.decode_batch(results[0], results[1], 0.5, 200)
        results = self.encoder.decode_batch(results[0], results[1], 0.5, 200)
        # batch size
        bs = len(results)
        for idx in range(0, bs):
            processed_results.append([])
            detection_boxes = results[idx][0]
            detection_classes = results[idx][1]
            expected_classes = expected[idx][0]
            scores = results[idx][2]
            for detection in range(0, len(scores)):
                if scores[detection] < 0.05:
                    break
                detection_class = int(detection_classes[detection])
                if detection_class in expected_classes:
                    self.good += 1
                box = detection_boxes[detection]
                # comes from model as:  0=xmax 1=ymax 2=xmin 3=ymin
                processed_results[idx].append([float(ids[idx]),
                                              box[1], box[0], box[3], box[2],
                                              scores[detection],
                                              float(detection_class)])
                self.total += 1
        return processed_results

class PostProcessCocoTfNative(PostProcessCoco):
    """
    Post processing required by ssd-resnet34 / pytorch
    """
    def __init__(self):
        super().__init__()
        self.use_inv_map = True

    def __call__(self, results, ids, expected=None, result_dict=None):
        # results come as:
        #   detection_boxes,detection_classes,detection_scores

        processed_results = []
        # batch size
        bs = len(results[0])
        for idx in range(0, bs):
            processed_results.append([])
            detection_boxes = results[0][idx]
            detection_classes = results[1][idx]
            expected_classes = expected[idx][0]
            scores = results[2][idx]
            for detection in range(0, len(scores)):
                if scores[detection] < 0.05:
                    break
                detection_class = int(detection_classes[detection])
                if detection_class in expected_classes:
                    self.good += 1
                box = detection_boxes[detection]
                # comes from model as:  0=xmax 1=ymax 2=xmin 3=ymin
                processed_results[idx].append([float(ids[idx]),
                                              box[0], box[1], box[2], box[3],
                                              scores[detection],
                                              float(detection_class)])
                self.total += 1
        return processed_results
