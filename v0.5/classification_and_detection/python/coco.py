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
        out_shape = [1200,1200]
        anchor_creator = self.AnchorCreator(out_shape,
                layers_shapes = [(50,50),(25,25),(13,13),(7,7),(3,3),(1,1)],
                anchor_scales = [(0.1,),(0.2,),(0.375,),(0.55,), (0.725,),(0.9,)],
                extra_anchor_scales = [(0.1414,),(0.2739,),(0.4541,),(0.6315,),(0.8078,),(0.9836,)],
                anchor_ratios = [(1.,2.,.5),(1.,2.,3.,.5,0.3333),(1.,2.,3.,.5,0.3333), (1.,2.,3.,.5,0.3333), (1.,2.,.5),(1.,2.,.5)],
                layer_steps=[24,48,92,171,400,1200])

        dboxes = anchor_creator.get_all_anchors()
        self.encoder = self.Encoder(dboxes)

    class AnchorCreator(object):
        def __init__(self, img_shape, layers_shapes, anchor_scales, extra_anchor_scales, anchor_ratios, layer_steps):
            super(PostProcessCocoTf.AnchorCreator, self).__init__()
            # img_shape -> (height, width)
            self._img_shape = img_shape
            self._layers_shapes = layers_shapes
            self._anchor_scales = anchor_scales
            self._extra_anchor_scales = extra_anchor_scales
            self._anchor_ratios = anchor_ratios
            self._layer_steps = layer_steps
            self._anchor_offset = [0.5] * len(self._layers_shapes)

        def get_layer_anchors(self, layer_shape, anchor_scale, extra_anchor_scale, anchor_ratio, layer_step, offset = 0.5):
            from itertools import product
            import math
            xy_on_layer = np.array(list(product(range(layer_shape[1]), range(layer_shape[0]))), dtype='float32')
            x_on_image = (xy_on_layer[:,:1] + offset) * layer_step / self._img_shape[0]
            y_on_image = (xy_on_layer[:,1:] + offset) * layer_step / self._img_shape[1]

            list_h_on_image = []
            list_w_on_image = []

            global_index = 0
            # for square anchors
            for _, scale in enumerate(extra_anchor_scale):
                list_h_on_image.append(scale)
                list_w_on_image.append(scale)
                global_index += 1
            # for other aspect ratio anchors
            for scale_index, scale in enumerate(anchor_scale):
                for ratio_index, ratio in enumerate(anchor_ratio):
                    list_h_on_image.append(scale / math.sqrt(ratio))
                    list_w_on_image.append(scale * math.sqrt(ratio))
                    global_index += 1
            h_on_image = np.array(list_h_on_image, dtype='float32')
            w_on_image = np.array(list_w_on_image, dtype='float32')
            ymin, xmin, ymax, xmax = self.center2point(y_on_image, x_on_image, h_on_image, w_on_image)
            ymin = ymin.reshape(ymin.size, -1)
            xmin = xmin.reshape(xmin.size, -1)
            ymax = ymax.reshape(ymax.size, -1)
            xmax = xmax.reshape(xmax.size, -1)
            xc, yc, h, w = self.point2center(ymin, xmin, ymax, xmax)

            return np.concatenate([xc,yc, w, h], axis=1)

        def center2point(self, center_y, center_x, height, width):
            return center_y - height / 2., center_x - width / 2., center_y + height / 2., center_x + width / 2.,

        def point2center(self, ymin, xmin, ymax, xmax):
            height, width = (ymax - ymin), (xmax - xmin)
            return ymin + height / 2., xmin + width / 2., height, width

        def get_all_anchors(self):
            all_anchors = []
            all_num_anchors_depth = []
            all_num_anchors_spatial = []
            for layer_index, layer_shape in enumerate(self._layers_shapes):
                anchors_this_layer = self.get_layer_anchors(layer_shape,
                    self._anchor_scales[layer_index],
                    self._extra_anchor_scales[layer_index],
                    self._anchor_ratios[layer_index],
                    self._layer_steps[layer_index],
                    self._anchor_offset[layer_index])
                all_anchors.append(anchors_this_layer)

            return np.vstack(all_anchors)


    class Encoder(object):
        def __init__(self, dboxes, scale_xy = 0.1, scale_wh = 0.2):
            self.dboxes = np.expand_dims(dboxes,0)
            self.nboxes = self.dboxes.shape[0]
            self.scale_xy = scale_xy
            self.scale_wh = scale_wh

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
            bboxes_in[:, :, [0,1,2,3]] = bboxes_in[:, :, [1,0,3,2]]
            bboxes_in[:, :, :2] = self.scale_xy*bboxes_in[:, :, :2]
            bboxes_in[:, :, 2:] = self.scale_wh*bboxes_in[:, :, 2:]

            bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*self.dboxes[:, :, 2:] + self.dboxes[:, :, :2]
            bboxes_in[:, :, 2:] = np.exp(bboxes_in[:, :, 2:])*self.dboxes[:, :, 2:]

            # Transform format to ltrb
            l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                         bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                         bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                         bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

            l = np.maximum(l, 0)
            t = np.maximum(t, 0)
            r = np.minimum(r, 1)
            b = np.minimum(b, 1)

            bboxes_in[:, :, 0] = l
            bboxes_in[:, :, 1] = t
            bboxes_in[:, :, 2] = r
            bboxes_in[:, :, 3] = b

            return bboxes_in, self.softmax_cpu(scores_in, dim=-1)

        def decode_batch(self, bboxes_in, scores_in,  criteria = 0.45, max_output=200):
            bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)
            output = []
            for bbox, prob in zip(bboxes, probs):
                output.append(self.decode_single(bbox, prob, criteria, max_output))
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
        processed_results = []
        results = self.encoder.decode_batch(results[0], results[1], 0.45, 200)
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


