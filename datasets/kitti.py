# Copyright (c) Facebook, Inc. and its affiliates.


""" 
Modified from https://github.com/facebookresearch/votenet
Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

-----

Again modified by: Ryan Glaspey
Date: 2022
"""
from cmath import pi
import os
import sys
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio  # to load .mat files for depth points

import utils.pc_util as pc_util
from utils.random_cuboid import RandomCuboid
from utils.pc_util import rotz, shift_scale_points, scale_points
from utils.box_util import (
    flip_axis_to_camera_tensor,
    get_3d_box_batch_tensor,
    flip_axis_to_camera_np,
    get_3d_box_batch_np,
)
from datasets.kitti_util import Calibration, compute_box_3d, load_velo_scan, read_label


MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1
DATA_PATH_V1 = "" ## Replace with path to dataset
DATA_PATH_V2 = "" ## Not used in the codebase.

# refs https://stackoverflow.com/a/13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class KITTI3DObjectDetectionDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 3
        self.num_angle_bin = 12
        self.max_num_obj = 64
        self.type2class = {
            "Car": 0,
            "Pedestrian": 1,
            "Cyclist": 2,
        }
        # self.type2class = {
        #     "DontCare": 0, #TODO should this be 0?
        #     "Car": 1,
        #     "Van": 2,
        #     "Truck": 3,
        #     "Pedestrian": 4,
        #     "Person_sitting": 5,
        #     "Cyclist": 6,
        #     "Tram": 7,
        #     "Misc": 8,
        # }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        # self.type2onehotclass = {
        #     "DontCare": 0, #TODO should this be 0?
        #     "Car": 1,
        #     "Van": 2,
        #     "Truck": 3,
        #     "Pedestrian": 4,
        #     "Person_sitting": 5,
        #     "Cyclist": 6,
        #     "Tram": 7,
        #     "Misc": 8,
        # }
        self.type2onehotclass = {
            "Car": 0,
            "Pedestrian": 1,
            "Cyclist": 2,
        }

    def angle2class(self, angle):
        """Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        returns class [0,1,...,N-1] and a residual number such that
            class*(2pi/N) + number = angle
        """
        num_class = self.num_angle_bin
        angle = angle % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (
            class_id * angle_per_class + angle_per_class / 2
        )
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        """Inverse function to angle2class"""
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2angle_batch(pred_cls, residual, to_label_format)

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    def my_compute_box_3d(self, center, size, heading_angle):
        R = pc_util.rotz(-1 * heading_angle)
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)

#TODO data augmentation, add transforms param? naw just do it inline
class KITTI3DObjectDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        augment=False,
        num_points=20000,
        # clip_size=4,
    ):
        assert num_points <= 50000
        # assert augment == False
        assert split_set in ["train", "val"]
        self.dataset_config = dataset_config

        assert root_dir != None

        # assert clip_size >= 1

        # self.clip_size = clip_size

        #TODO refactor and improve splitting
        self.root_dir = root_dir
        self.data_path = os.path.join(root_dir, "training")
        all_ids = [(f[:-len('.txt')], idx) for idx, f in enumerate(os.listdir(os.path.join(self.data_path, 'velodyne')))]
        self.ids = all_ids[:4000] if split_set == 'train' else all_ids[4000:]

        # Copy the raw KITTI dataset into a /raw folder of the KITTI Object Detection benchmark
        # indices_for_mappings = None
        # with open(os.path.join(root_dir, 'mapping', 'train_rand.txt'), 'r') as inp:
        #     indices_for_mappings = [int(value)-1 for value in inp.read().split(',')]
        # raw_mapping_lines = None
        # with open(os.path.join(root_dir, 'mapping', 'train_mapping.txt'), 'r') as inp:
        #     raw_mapping_lines = [line.split(' ') for line in inp]
        # mappings_to_raw = [(os.path.join(root_dir, 'raw', line[0], f'{line[1]}', 'velodyne_points', 'data'), int(line[2])) for line in raw_mapping_lines]
        # self.raw_mapping = [mappings_to_raw[index] for index in indices_for_mappings]

        self.num_points = num_points
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = 64

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        string_id, raw_mapping_id = self.ids[idx]

        point_cloud = load_velo_scan(os.path.join(self.data_path, 'velodyne', f'{string_id}.bin'))[:, 0:3]
        calib = Calibration(os.path.join(self.data_path, 'calib', f'{string_id}.txt'))
        objects = read_label(os.path.join(self.data_path, 'label_2', f'{string_id}.txt'))
        # Only use objects of classes with enough data
        objects = [object for object in objects if object.type in ['Car', 'Pedestrian', 'Cyclist']]

        # raw_mapping_dir, raw_mapping_id = self.raw_mapping[raw_mapping_id]
        # point_cloud_clip = [load_velo_scan(os.path.join(raw_mapping_dir, f'{max(0, raw_mapping_id-i):010d}.bin')) for i in range(self.clip_size)]
        # for point_cloud_i in point_cloud_clip:
        #     print(point_cloud_i.shape)
        # point_cloud = point_cloud_clip[0]

        bboxes = []
        _og_bboxes = []
        _intermediate_boxes = []
        for idx, object in enumerate(objects):
            box3d_pts_3d = compute_box_3d(object, calib.P)[1]
            box3d_pts_3d = calib.project_rect_to_velo(box3d_pts_3d)
            _og_bboxes.append(box3d_pts_3d.copy())

            v2 = box3d_pts_3d[1][:2]
            v1 = box3d_pts_3d[0][:2]
            angle = angle_between(v2 - v1, np.array([1, 0]))
            # cross product is positive if v2 is closer clockwise than counterclockwise from v1. I'm pretty sure anyways.
            if np.cross(v1, v2) > 0: angle = -angle

            label_center = np.mean(box3d_pts_3d, axis=0)
            box3d_pts_3d -= label_center
            box3d_pts_3d = np.matmul(box3d_pts_3d, rotz(-angle))
            box3d_pts_3d += label_center

            _intermediate_boxes.append(box3d_pts_3d.copy())

            new_bbox = np.array([
                np.min(box3d_pts_3d[:,0]), np.min(box3d_pts_3d[:,1]), np.min(box3d_pts_3d[:,2]), 
                np.max(box3d_pts_3d[:,0]), np.max(box3d_pts_3d[:,1]), np.max(box3d_pts_3d[:,2]),
            ])

            width, length, height = (new_bbox[3] - new_bbox[0]) / 2, (new_bbox[4] - new_bbox[1]) / 2, (new_bbox[5] - new_bbox[2]) / 2
            bboxes.append(np.array([
                new_bbox[0] + width, new_bbox[1] + length, new_bbox[2] + height,
                width, length, height,
                angle,
                self.dataset_config.type2class[object.type],
            ]))

        bboxes = np.array(bboxes)

        # ------------------------------- LABELS ------------------------------
        angle_classes = np.zeros((self.max_num_obj,), dtype=np.float32)
        angle_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_angles = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_sizes = np.zeros((self.max_num_obj, 3), dtype=np.float32)
        label_mask = np.zeros((self.max_num_obj))
        label_mask[0 : bboxes.shape[0]] = 1
        max_bboxes = np.zeros((self.max_num_obj, 8))
        max_bboxes[0 : bboxes.shape[0], :] = bboxes

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((self.max_num_obj, 6))

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            raw_angles[i] = bbox[6] % 2 * np.pi
            box3d_size = bbox[3:6] * 2
            raw_sizes[i, :] = box3d_size
            angle_class, angle_residual = self.dataset_config.angle2class(bbox[6])
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            corners_3d = self.dataset_config.my_compute_box_3d(
                bbox[0:3], bbox[3:6], bbox[6]
            )
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2,
                    xmax - xmin,
                    ymax - ymin,
                    zmax - zmin,
                ]
            )
            target_bboxes[i, :] = target_bbox

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )

        point_cloud_dims_min = point_cloud.min(axis=0)
        point_cloud_dims_max = point_cloud.max(axis=0)

        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]

        # re-encode angles to be consistent with VoteNet eval
        angle_classes = angle_classes.astype(np.int64)
        angle_residuals = angle_residuals.astype(np.float32)
        raw_angles = self.dataset_config.class2angle_batch(
            angle_classes, angle_residuals
        )

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        target_bboxes_semcls = np.zeros((self.max_num_obj))
        target_bboxes_semcls[0 : bboxes.shape[0]] = bboxes[:, -1]  # from 0 to 8
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes
        ret_dict["gt_angle_residual_label"] = angle_residuals
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max

        #TODO remove
        # ret_dict["_testing"] = [self.dataset_config.my_compute_box_3d(
        #         bbox[0:3], bbox[3:6], bbox[6]
        #     ) for bbox in bboxes]
        # ret_dict["_verification"] = _og_bboxes
        # ret_dict["_intermediate"] = _intermediate_boxes
        return ret_dict
