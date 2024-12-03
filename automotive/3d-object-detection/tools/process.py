import shapely.geometry
import numpy as np
import torch
import copy


def bbox_camera2lidar(bboxes, tr_velo_to_cam, r0_rect):
    '''
    bboxes: shape=(N, 7)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    '''
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)
    extended_xyz = np.pad(
        bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    bboxes_lidar = np.concatenate(
        [xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return np.array(bboxes_lidar, dtype=np.float32)


def bbox_lidar2camera(bboxes, tr_velo_to_cam, r0_rect):
    '''
    bboxes: shape=(N, 7)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    '''
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = torch.cat([y_size, z_size, x_size], axis=1)
    extended_xyz = torch.nn.functional.pad(
        bboxes[:, :3], (0, 1), 'constant', value=1.0)
    rt_mat = r0_rect @ tr_velo_to_cam
    xyz = extended_xyz @ rt_mat.T
    bboxes_camera = torch.cat([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return bboxes_camera


def bbox3d2corners_camera(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
        z (front)            6 ------ 5
        /                  / |     / |
       /                  2 -|---- 1 |
      /                   |  |     | |
    |o ------> x(right)   | 7 -----| 4
    |                     |/   o   |/
    |                     3 ------ 0
    |
    v y(down)
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = torch.tensor([[0.5, 0.0, -0.5], [0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5, 0.0, -0.5],
                                   [0.5, 0.0, 0.5], [0.5, -1.0, 0.5], [-0.5, -1.0, 0.5], [-0.5, 0.0, 0.5]])
    # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :]

    # 2. rotate around y axis
    rot_sin, rot_cos = torch.sin(angles), torch.cos(angles)
    # in fact, angle
    rot_mat = torch.stack([torch.stack([rot_cos, torch.zeros_like(rot_cos), rot_sin]),
                           torch.stack([torch.zeros_like(rot_cos), torch.ones_like(
                               rot_cos), torch.zeros_like(rot_cos)]),
                           torch.stack([-rot_sin, torch.zeros_like(rot_cos), rot_cos])])  # (3, 3, n)
    rot_mat = torch.permute(rot_mat, (2, 1, 0))  # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat  # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners.clone().detach()


def points_camera2image(points, P2):
    '''
    points: shape=(N, 8, 3)
    P2: shape=(4, 4)
    return: shape=(N, 8, 2)
    '''
    extended_points = torch.nn.functional.pad(
        points, (0, 1), 'constant', value=1.0)  # (n, 8, 4)
    image_points = extended_points @ P2.T  # (N, 8, 4)
    image_points = image_points[:, :, :2] / image_points[:, :, 2:3]
    return image_points.clone().detach()


def keep_bbox_from_image_range(
        result, calib_info, num_images, image_info, cam_sync=False):
    r0_rect = calib_info['R0_rect']
    lidar_bboxes = result['lidar_bboxes']
    labels = result['labels']
    scores = result['scores']
    total_keep_flag = torch.zeros(lidar_bboxes.size(dim=0)).bool()
    for i in range(num_images):
        h, w = image_info['camera'][i]['image_shape']
        tr_velo_to_cam = calib_info['Tr_velo_to_cam_' + str(i)]
        P = calib_info['P' + str(i)]
        camera_bboxes = bbox_lidar2camera(
            lidar_bboxes, tr_velo_to_cam, r0_rect)  # (n, 7)
        if i == 0:
            main_camera_bboxes = camera_bboxes.clone()
        bboxes_points = bbox3d2corners_camera(camera_bboxes)  # (n, 8, 3)
        image_points = points_camera2image(bboxes_points, P)  # (n, 8, 2)
        image_x1y1 = torch.min(image_points, axis=1)[0]  # (n, 2)
        image_x1y1 = torch.maximum(image_x1y1, torch.tensor(0))
        image_x2y2 = torch.max(image_points, axis=1)[0]  # (n, 2)
        image_x2y2 = torch.minimum(image_x2y2, torch.tensor([w, h]))
        bboxes2d = torch.cat([image_x1y1, image_x2y2], axis=-1)

        keep_flag = (image_x1y1[:, 0] < w) & (image_x1y1[:, 1] < h) & (
            image_x2y2[:, 0] > 0) & (image_x2y2[:, 1] > 0) & (camera_bboxes[:, 2] > 0)
        total_keep_flag = total_keep_flag | keep_flag
    if cam_sync:
        result = {
            'lidar_bboxes': lidar_bboxes[total_keep_flag],
            'labels': labels[total_keep_flag],
            'scores': scores[total_keep_flag],
            'bboxes2d': bboxes2d[total_keep_flag],
            'camera_bboxes': main_camera_bboxes[total_keep_flag]
        }
    else:
        result = {
            'lidar_bboxes': lidar_bboxes,
            'labels': labels,
            'scores': scores,
            'bboxes2d': bboxes2d,
            'camera_bboxes': main_camera_bboxes
        }
    return result


def limit_period(val, offset=0.5, period=np.pi):
    """
    val: array or float
    offset: float
    period: float
    return: Value in the range of [-offset * period, (1-offset) * period]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def iou2d(bboxes1, bboxes2, metric=0):
    '''
    bboxes1: (n, 4), (x1, y1, x2, y2)
    bboxes2: (m, 4), (x1, y1, x2, y2)
    return: (n, m)
    '''
    rows = len(bboxes1)
    cols = len(bboxes2)
    if rows * cols == 0:
        return torch.empty((rows, cols))
    bboxes_x1 = torch.maximum(
        bboxes1[:, 0][:, None], bboxes2[:, 0][None, :])  # (n, m)
    bboxes_y1 = torch.maximum(
        bboxes1[:, 1][:, None], bboxes2[:, 1][None, :])  # (n, m)
    bboxes_x2 = torch.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    bboxes_y2 = torch.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])

    bboxes_w = torch.clamp(bboxes_x2 - bboxes_x1, min=0)
    bboxes_h = torch.clamp(bboxes_y2 - bboxes_y1, min=0)

    iou_area = bboxes_w * bboxes_h  # (n, m)

    bboxes1_wh = bboxes1[:, 2:] - bboxes1[:, :2]
    area1 = bboxes1_wh[:, 0] * bboxes1_wh[:, 1]  # (n, )
    bboxes2_wh = bboxes2[:, 2:] - bboxes2[:, :2]
    area2 = bboxes2_wh[:, 0] * bboxes2_wh[:, 1]  # (m, )
    if metric == 0:
        iou = iou_area / (area1[:, None] + area2[None, :] - iou_area + 1e-8)
    elif metric == 1:
        iou = iou_area / (area1[:, None] + 1e-8)
    return iou


def nearest_bev(bboxes):
    '''
    bboxes: (n, 7), (x, y, z, w, l, h, theta)
    return: (n, 4), (x1, y1, x2, y2)
    '''
    bboxes_bev = copy.deepcopy(bboxes[:, [0, 1, 3, 4]])
    bboxes_angle = limit_period(
        bboxes[:, 6].cpu(), offset=0.5, period=np.pi).to(bboxes_bev)
    bboxes_bev = torch.where(torch.abs(
        bboxes_angle[:, None]) > np.pi / 4, bboxes_bev[:, [0, 1, 3, 2]], bboxes_bev)

    bboxes_xy = bboxes_bev[:, :2]
    bboxes_wl = bboxes_bev[:, 2:]
    bboxes_bev_x1y1x2y2 = torch.cat(
        [bboxes_xy - bboxes_wl / 2, bboxes_xy + bboxes_wl / 2], dim=-1)
    return bboxes_bev_x1y1x2y2


def iou2d_nearest(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7),
    return: (n, m)
    '''
    bboxes1_bev = nearest_bev(bboxes1)
    bboxes2_bev = nearest_bev(bboxes2)
    iou = iou2d(bboxes1_bev, bboxes2_bev)
    return iou


def limit_period(val, offset=0.5, period=np.pi):
    """
    val: array or float
    offset: float
    period: float
    return: Value in the range of [-offset * period, (1-offset) * period]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def iou3d_camera(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7)
    return: (n, m)
    '''
    rows = len(bboxes1)
    cols = len(bboxes2)
    if rows * cols == 0:
        return torch.empty((rows, cols))
    # 1. height overlap
    bboxes1_bottom, bboxes2_bottom = bboxes1[:, 1] - \
        bboxes1[:, 4], bboxes2[:, 1] - bboxes2[:, 4]  # (n, ), (m, )
    bboxes1_top, bboxes2_top = bboxes1[:, 1], bboxes2[:, 1]  # (n, ), (m, )
    bboxes_bottom = torch.maximum(
        bboxes1_bottom[:, None], bboxes2_bottom[None, :])  # (n, m)
    bboxes_top = torch.minimum(bboxes1_top[:, None], bboxes2_top[None, :])
    height_overlap = torch.clamp(bboxes_top - bboxes_bottom, min=0)

    # 2. bev overlap
    bboxes1_x1y1 = bboxes1[:, [0, 2]] - bboxes1[:, [3, 5]] / 2
    bboxes1_x2y2 = bboxes1[:, [0, 2]] + bboxes1[:, [3, 5]] / 2
    bboxes2_x1y1 = bboxes2[:, [0, 2]] - bboxes2[:, [3, 5]] / 2
    bboxes2_x2y2 = bboxes2[:, [0, 2]] + bboxes2[:, [3, 5]] / 2
    bboxes1_bev = torch.cat(
        [bboxes1_x1y1, bboxes1_x2y2, bboxes1[:, 6:]], dim=-1)
    bboxes2_bev = torch.cat(
        [bboxes2_x1y1, bboxes2_x2y2, bboxes2[:, 6:]], dim=-1)
    bev_overlap = (
        rotated_box_iou(
            bboxes1_bev,
            bboxes2_bev)).to(
        device=height_overlap.device)  # (n, m)

    # 3. overlap and volume
    overlap = height_overlap * bev_overlap
    volume1 = bboxes1[:, 3] * bboxes1[:, 4] * bboxes1[:, 5]
    volume2 = bboxes2[:, 3] * bboxes2[:, 4] * bboxes2[:, 5]
    volume = volume1[:, None] + volume2[None, :]  # (n, m)

    # 4. iou
    iou = overlap / (volume - overlap + 1e-8)

    return iou


def boxes_overlap_bev(boxes_a, boxes_b):
    """Calculate boxes Overlap in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_overlap (torch.Tensor): Overlap result with shape (M, N).
    """
    ans_overlap = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    if ans_overlap.size(0) * ans_overlap.size(1) == 0:
        return ans_overlap
    boxes_overlap_bev_gpu(
        boxes_a.contiguous(),
        boxes_b.contiguous(),
        ans_overlap)

    return ans_overlap


def rotated_box_iou(boxes1, boxes2):
    """
    Calculates IoU for rotated bounding boxes.

    Args:
        boxes1 (torch.Tensor): Tensor of shape (N, 5) representing rotated boxes in format (x_center, y_center, width, height, angle).
        boxes2 (torch.Tensor): Tensor of shape (M, 5) representing rotated boxes in the same format.

    Returns:
        torch.Tensor: IoU matrix of shape (N, M).
    """

    # Convert boxes to polygons
    polygons1 = boxes_to_polygons(boxes1)
    polygons2 = boxes_to_polygons(boxes2)

    # Calculate IoU for each pair of polygons
    ious = torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    overlaps = torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(boxes1.shape[0]):
        for j in range(boxes2.shape[0]):
            intersection = polygon_intersection(polygons1[i], polygons2[j])
            union = polygon_union(polygons1[i], polygons2[j])
            ious[i, j] = intersection / union
            overlaps[i, j] = intersection

    return overlaps


def boxes_to_polygons(boxes):
    # Implementation to convert boxes to polygons
    polygons = []
    for box in boxes:
        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]
        polygon = shapely.geometry.Polygon(
            [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
        polygon = shapely.affinity.rotate(
            polygon, -1 * box[4], use_radians=True)
        polygons.append(polygon)
    return polygons


def polygon_intersection(polygon1, polygon2):
    return shapely.intersection(polygon1, polygon2).area


def polygon_union(polygon1, polygon2):
    # Implementation to calculate union area of polygons
    return shapely.union(polygon1, polygon2).area
