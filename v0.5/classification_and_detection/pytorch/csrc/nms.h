// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const double threshold) {

  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    if (dets.numel() == 0)
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
    return nms_cuda(b, threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  at::Tensor result = nms_cpu(dets, scores, threshold);
  return result;
}


at::Tensor multi_label_nms(const at::Tensor& boxes,
                           const at::Tensor& scores,
                           const at::Tensor& max_output_boxes_per_class,
                           const at::Tensor& iou_threshold,
                           const at::Tensor& score_threshold) {
  at::Tensor result = multi_label_nms_cpu(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);
  return result;
}
