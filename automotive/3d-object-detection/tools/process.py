import numpy as np
import torch

def bbox_camera2lidar(bboxes, tr_velo_to_cam, r0_rect):
    '''
    bboxes: shape=(N, 7)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    '''
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    bboxes_lidar = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
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
    extended_xyz = torch.nn.functional.pad(bboxes[:, :3], (0, 1), 'constant', value=1.0)
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
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around y axis
    rot_sin, rot_cos = torch.sin(angles), torch.cos(angles)
    # in fact, angle
    rot_mat = torch.stack([torch.stack([rot_cos, torch.zeros_like(rot_cos), rot_sin]),
                        torch.stack([torch.zeros_like(rot_cos), torch.ones_like(rot_cos), torch.zeros_like(rot_cos)]),
                        torch.stack([-rot_sin, torch.zeros_like(rot_cos), rot_cos])]) # (3, 3, n)
    rot_mat = torch.permute(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners.clone().detach()

def points_camera2image(points, P2):
    '''
    points: shape=(N, 8, 3) 
    P2: shape=(4, 4)
    return: shape=(N, 8, 2)
    '''
    extended_points = torch.nn.functional.pad(points, (0, 1), 'constant', value=1.0) # (n, 8, 4)
    image_points = extended_points @ P2.T # (N, 8, 4)
    image_points = image_points[:, :, :2] / image_points[:, :, 2:3]
    return image_points.clone().detach()

def keep_bbox_from_image_range(result, calib_info, num_images, image_info, cam_sync=False):
    r0_rect = calib_info['R0_rect']
    lidar_bboxes = result['lidar_bboxes']
    labels = result['labels']
    scores = result['scores']
    total_keep_flag = torch.zeros(lidar_bboxes.size(dim=0)).bool()
    for i in range(num_images):
        h, w = image_info['camera'][i]['image_shape']
        tr_velo_to_cam = calib_info['Tr_velo_to_cam_' + str(i)]
        P = calib_info['P' + str(i)]
        camera_bboxes = bbox_lidar2camera(lidar_bboxes, tr_velo_to_cam, r0_rect) # (n, 7)
        if i == 0:
            main_camera_bboxes = camera_bboxes.clone()
        bboxes_points = bbox3d2corners_camera(camera_bboxes) # (n, 8, 3)
        image_points = points_camera2image(bboxes_points, P) # (n, 8, 2)
        image_x1y1 = torch.min(image_points, axis=1)[0] # (n, 2)
        image_x1y1 = torch.maximum(image_x1y1, torch.tensor(0))
        image_x2y2 = torch.max(image_points, axis=1)[0] # (n, 2)
        image_x2y2 = torch.minimum(image_x2y2, torch.tensor([w, h]))
        bboxes2d = torch.cat([image_x1y1, image_x2y2], axis=-1)

        keep_flag = (image_x1y1[:, 0] < w) & (image_x1y1[:, 1] < h) & (image_x2y2[:, 0] > 0) & (image_x2y2[:, 1] > 0) & (camera_bboxes[:, 2] > 0)
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
        result =  {
            'lidar_bboxes': lidar_bboxes,
            'labels': labels,
            'scores': scores,
            'bboxes2d': bboxes2d,
            'camera_bboxes': main_camera_bboxes
        }
    return result