// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "cpu/vision.h"


template <typename scalar_t>
at::Tensor nms_cpu_kernel(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= threshold)
        suppressed[j] = 1;
   }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}

at::Tensor nms_cpu(const at::Tensor& dets,
               const at::Tensor& scores,
               const double threshold) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms", [&] {
    result = nms_cpu_kernel<scalar_t>(dets, scores, threshold);
  });
  return result;
}

template <typename scalar_t>
at::Tensor multi_label_nms_cpu_kernel(const at::Tensor& boxes,
                                      const at::Tensor& scores,
                                      const at::Tensor& max_output_boxes_per_class,
                                      const at::Tensor& iou_threshold,
                                      const at::Tensor& score_threshold) {
  AT_ASSERTM(!boxes.type().is_cuda(), "boxes must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(boxes.type() == scores.type(), "boxes should have the same type as scores");
  AT_ASSERTM(boxes.size(0) == 1, "only support batch size = 1.");

  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong).device(at::kCPU));
  }


  auto max_output_per_class = max_output_boxes_per_class.data<int64_t>()[0];
  auto iou_thres = iou_threshold.data<scalar_t>()[0];
  auto score_thres = score_threshold.data<scalar_t>()[0];
  //printf("score threshold is %f\n", float(score_thres));

  auto dets = boxes.squeeze(0);
  auto single_batch_scores = scores.squeeze(0);

  auto nlabels = single_batch_scores.size(0);

  std::vector<at::Tensor> res_tensors;
  for (int64_t _l = 0; _l < nlabels; _l++){
    auto lscores_t = single_batch_scores.select(0, _l).contiguous();
    auto lscores = lscores_t.data<scalar_t>();

    auto x1_t = dets.select(1, 0).contiguous();
    auto y1_t = dets.select(1, 1).contiguous();
    auto x2_t = dets.select(1, 2).contiguous();
    auto y2_t = dets.select(1, 3).contiguous();

    at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

    auto order_t = std::get<1>(lscores_t.sort(0, /* descending=*/true));

    auto ndets = dets.size(0);
    at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

    auto suppressed = suppressed_t.data<uint8_t>();
    auto order = order_t.data<int64_t>();
    auto x1 = x1_t.data<scalar_t>();
    auto y1 = y1_t.data<scalar_t>();
    auto x2 = x2_t.data<scalar_t>();
    auto y2 = y2_t.data<scalar_t>();
    auto areas = areas_t.data<scalar_t>();

    int64_t nleft = 0;
    for (int64_t _i = 0; _i < ndets; _i++) {
      auto i = order[_i];
      if (lscores[i] <= score_thres || nleft >= max_output_per_class) {
        suppressed[i] = 1;
        continue;
      }
      // Add count here to match behavior of original SSD model in this repo.
      // nleft++;
      if (suppressed[i] == 1)
        continue;
      auto ix1 = x1[i];
      auto iy1 = y1[i];
      auto ix2 = x2[i];
      auto iy2 = y2[i];
      auto iarea = areas[i];

      for (int64_t _j = _i + 1; _j < ndets; _j++) {
        auto j = order[_j];
        if (suppressed[j] == 1)
          continue;
        auto xx1 = std::max(ix1, x1[j]);
        auto yy1 = std::max(iy1, y1[j]);
        auto xx2 = std::min(ix2, x2[j]);
        auto yy2 = std::min(iy2, y2[j]);

        auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
        auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
        auto inter = w * h;
        auto ovr = inter / (iarea + areas[j] - inter);
        if (ovr >= iou_thres)
          suppressed[j] = 1;
      }
      // Add count here to match behavior of ONNXRuntime NMS implementation.
      nleft++;
    }
    auto selected_idx = at::nonzero(suppressed_t == 0).squeeze(1);
    auto nselected = selected_idx.size(0);
    auto batch_idx = at::zeros({nselected}, dets.options().dtype(at::kLong).device(at::kCPU));
    auto label_idx = at::full({nselected}, _l, dets.options().dtype(at::kLong).device(at::kCPU));
    auto result_tensor = at::cat({batch_idx.unsqueeze(1), label_idx.unsqueeze(1), selected_idx.unsqueeze(1)}, 1);
    res_tensors.push_back(result_tensor);
  }
  auto result = at::cat(res_tensors, 0);
  return result;
}



at::Tensor multi_label_nms_cpu(const at::Tensor& boxes,
                               const at::Tensor& scores,
                               const at::Tensor& max_output_boxes_per_class,
                               const at::Tensor& iou_threshold,
                               const at::Tensor& score_threshold) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(boxes.scalar_type(), "multi_label_nms", [&] {
    result = multi_label_nms_cpu_kernel<scalar_t>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);
  });
  return result;
}

