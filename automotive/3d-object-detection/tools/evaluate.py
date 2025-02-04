import argparse
import numpy as np
import os
import torch
from tqdm import tqdm

from tools.process import iou2d, iou3d_camera


def get_score_thresholds(tp_scores, total_num_valid_gt, num_sample_pts=41):
    score_thresholds = []
    tp_scores = sorted(tp_scores)[::-1]
    cur_recall, pts_ind = 0, 0
    for i, score in enumerate(tp_scores):
        lrecall = (i + 1) / total_num_valid_gt
        rrecall = (i + 2) / total_num_valid_gt

        if i == len(tp_scores) - 1:
            score_thresholds.append(score)
            break

        if (lrecall + rrecall) / 2 < cur_recall:
            continue

        score_thresholds.append(score)
        pts_ind += 1
        cur_recall = pts_ind / (num_sample_pts - 1)
    return score_thresholds


def convert_calib(calib, cuda):
    result = {}
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    result['R0_rect'] = torch.from_numpy(
        calib['R0_rect']).to(
        device=device,
        dtype=torch.float)
    for i in range(5):
        result['P' + str(i)] = torch.from_numpy(calib['P' + str(i)]
                                                ).to(device=device, dtype=torch.float)
        result['Tr_velo_to_cam_' +
               str(i)] = torch.from_numpy(calib['Tr_velo_to_cam_' +
                                                str(i)]).to(device=device, dtype=torch.float)
    return result


def do_eval(det_results, gt_results, CLASSES, cam_sync=False):
    '''
    det_results: list,
    gt_results: dict(id -> det_results)
    CLASSES: dict
    '''

    # 1. calculate iou
    ious = {
        'bbox_3d': []
    }
    # ids = list(sorted([g['image']['image_idx'] for g in gt_results]))
    if cam_sync:
        annos_label = 'cam_sync_annos'
    else:
        annos_label = 'annos'
    for id in range(len(gt_results)):
        gt_result = gt_results[id][annos_label]
        if gt_results[id]['image']['image_idx'] not in det_results:
            continue
        det_result = det_results[gt_results[id]['image']['image_idx']]

        # 1.2, bev iou
        gt_location = gt_result['location'].astype(np.float32)
        gt_dimensions = gt_result['dimensions'].astype(np.float32)
        gt_rotation_y = gt_result['rotation_y'].astype(np.float32)
        det_location = det_result['location'].astype(np.float32).reshape(-1, 3)
        det_dimensions = det_result['dimensions'].astype(
            np.float32).reshape(-1, 3)
        det_rotation_y = det_result['rotation_y'].astype(np.float32)

        # 1.3, 3dbboxes iou
        gt_bboxes3d = np.concatenate(
            [gt_location, gt_dimensions, gt_rotation_y[:, None]], axis=-1)
        det_bboxes3d = np.concatenate(
            [det_location, det_dimensions, det_rotation_y[:, None]], axis=-1)
        iou3d_v = iou3d_camera(
            torch.from_numpy(gt_bboxes3d).to(
                "cuda" if torch.cuda.is_available() else "cpu"),
            torch.from_numpy(det_bboxes3d).to("cuda" if torch.cuda.is_available() else "cpu"))
        ious['bbox_3d'].append(iou3d_v.cpu().numpy())

    MIN_IOUS = {
        'Pedestrian': [0.5],
        'Cyclist': [0.5],
        'Car': [0.7]
    }
    MIN_HEIGHT = [-1]

    overall_results = {}
    for e_ind, eval_type in enumerate(['bbox_3d']):
        eval_ious = ious[eval_type]
        eval_ap_results, eval_aos_results = {}, {}
        for cls in CLASSES:
            eval_ap_results[cls] = []
            eval_aos_results[cls] = []
            CLS_MIN_IOU = MIN_IOUS[cls][e_ind]
            for difficulty in [0]:
                # 1. bbox property
                total_gt_ignores, total_det_ignores, total_dc_bboxes, total_scores = [], [], [], []
                for id in range(len(gt_results)):
                    gt_result = gt_results[id][annos_label]
                    det_result = det_results[gt_results[id]
                                             ['image']['image_idx']]

                    # 1.1 gt bbox property
                    cur_gt_names = gt_result['name']
                    cur_difficulty = gt_result['difficulty']
                    gt_ignores, dc_bboxes = [], []
                    for j, cur_gt_name in enumerate(cur_gt_names):
                        ignore = cur_difficulty[j] < 0 or cur_difficulty[j] > difficulty
                        if cur_gt_name == cls:
                            valid_class = 1
                        elif cls == 'Pedestrian' and cur_gt_name == 'Person_sitting':
                            valid_class = 0
                        elif cls == 'Car' and cur_gt_name == 'Van':
                            valid_class = 0
                        else:
                            valid_class = -1

                        if valid_class == 1 and not ignore:
                            gt_ignores.append(0)
                        elif valid_class == 0 or (valid_class == 1 and ignore):
                            gt_ignores.append(1)
                        else:
                            gt_ignores.append(-1)

                        if cur_gt_name == 'DontCare':
                            dc_bboxes.append(gt_result['bbox'][j])
                    total_gt_ignores.append(gt_ignores)
                    total_dc_bboxes.append(np.array(dc_bboxes))

                    # 1.2 det bbox property
                    cur_det_names = det_result['name']
                    if len(cur_det_names) == 0:
                        cur_det_heights = np.empty_like(det_result['bbox'])
                    else:
                        cur_det_heights = det_result['bbox'][:,
                                                             3] - det_result['bbox'][:, 1]
                    det_ignores = []
                    for j, cur_det_name in enumerate(cur_det_names):
                        if cur_det_heights[j] < MIN_HEIGHT[difficulty]:
                            det_ignores.append(1)
                        elif cur_det_name == cls:
                            det_ignores.append(0)
                        else:
                            det_ignores.append(-1)
                    total_det_ignores.append(det_ignores)
                    total_scores.append(det_result['score'])

                # 2. calculate scores thresholds for PR curve
                tp_scores = []
                for i in range(len(gt_results)):
                    cur_eval_ious = eval_ious[i]
                    gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                    scores = total_scores[i]

                    nn, mm = cur_eval_ious.shape
                    assigned = np.zeros((mm, ), dtype=np.bool_)
                    for j in range(nn):
                        if gt_ignores[j] == -1:
                            continue
                        match_id, match_score = -1, -1
                        for k in range(mm):
                            if not assigned[k] and det_ignores[k] >= 0 and cur_eval_ious[j,
                                                                                         k] > CLS_MIN_IOU and scores[k] > match_score:
                                match_id = k
                                match_score = scores[k]
                        if match_id != -1:
                            assigned[match_id] = True
                            if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                tp_scores.append(match_score)
                total_num_valid_gt = np.sum(
                    [np.sum(np.array(gt_ignores) == 0) for gt_ignores in total_gt_ignores])
                score_thresholds = get_score_thresholds(
                    tp_scores, total_num_valid_gt)

                # 3. draw PR curve and calculate mAP
                tps, fns, fps, total_aos = [], [], [], []

                for score_threshold in score_thresholds:
                    tp, fn, fp = 0, 0, 0
                    aos = 0
                    for i in range(len(gt_results)):
                        cur_eval_ious = eval_ious[i]
                        gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                        scores = total_scores[i]

                        nn, mm = cur_eval_ious.shape
                        assigned = np.zeros((mm, ), dtype=np.bool_)
                        for j in range(nn):
                            if gt_ignores[j] == -1:
                                continue
                            match_id, match_iou = -1, -1
                            for k in range(mm):
                                if not assigned[k] and det_ignores[k] >= 0 and scores[
                                        k] >= score_threshold and cur_eval_ious[j, k] > CLS_MIN_IOU:

                                    if det_ignores[k] == 0 and cur_eval_ious[j,
                                                                             k] > match_iou:
                                        match_iou = cur_eval_ious[j, k]
                                        match_id = k
                                    elif det_ignores[k] == 1 and match_iou == -1:
                                        match_id = k

                            if match_id != -1:
                                assigned[match_id] = True
                                if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                    tp += 1
                            else:
                                if gt_ignores[j] == 0:
                                    fn += 1

                        for k in range(mm):
                            if det_ignores[k] == 0 and scores[k] >= score_threshold and not assigned[k]:
                                fp += 1

                        # In case 2d bbox evaluation, we should consider
                        # dontcare bboxes
                        if eval_type == 'bbox_2d':
                            dc_bboxes = total_dc_bboxes[i]
                            det_bboxes = det_results[gt_results[i]
                                                     ['image']['image_idx']]['bbox']
                            if len(dc_bboxes) > 0:
                                ious_dc_det = iou2d(
                                    torch.from_numpy(det_bboxes),
                                    torch.from_numpy(dc_bboxes),
                                    metric=1).numpy().T
                                for j in range(len(dc_bboxes)):
                                    for k in range(len(det_bboxes)):
                                        if det_ignores[k] == 0 and scores[k] >= score_threshold and not assigned[k]:
                                            if ious_dc_det[j, k] > CLS_MIN_IOU:
                                                fp -= 1
                                                assigned[k] = True

                    tps.append(tp)
                    fns.append(fn)
                    fps.append(fp)
                    if eval_type == 'bbox_2d':
                        total_aos.append(aos)

                tps, fns, fps = np.array(tps), np.array(fns), np.array(fps)

                recalls = tps / (tps + fns)
                precisions = tps / (tps + fps)
                for i in range(len(score_thresholds)):
                    precisions[i] = np.max(precisions[i:])

                sums_AP = 0
                for i in range(0, len(score_thresholds), 4):
                    sums_AP += precisions[i]
                mAP = sums_AP / 11 * 100
                eval_ap_results[cls].append(mAP)
    return eval_ap_results
