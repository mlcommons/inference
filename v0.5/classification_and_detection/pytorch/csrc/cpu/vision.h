// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
/* Modifications Copyright (c) Microsoft. */
#pragma once
#include <torch/script.h>


at::Tensor nms_cpu(const at::Tensor& dets,
                   const at::Tensor& scores,
                   const double threshold);

at::Tensor multi_label_nms_cpu(const at::Tensor& boxes,
                               const at::Tensor& scores,
                               const at::Tensor& max_output_boxes_per_class,
                               const at::Tensor& iou_threshold,
                               const at::Tensor& score_threshold);
