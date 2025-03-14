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
from datasets.kitti_raw_mapping import KittiRawMapping
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio  # to load .mat files for depth points
from sklearn.model_selection import train_test_split
import tarfile
from utils.axis_align_util import convert_box_corners_into_obb

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

class KITTI3DObjectDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        use_height=False,
        augment=False,
        num_points=20000,
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
    ):
        assert num_points <= 50000
        assert split_set in ["train", "val", "train-clip", "val-clip", "video"]
        self.dataset_config = dataset_config
        self.split_set = split_set

        assert root_dir != None

        self.root_dir = root_dir

        self.raw_mapper = \
            KittiRawMapping(root_dir) if self.split_set in ["train-clip", "val-clip", "video"] else None

        self.data_len = 0
        self.data_path = None
        self.data_path_video = None
        self.data_path_video_calib = None
        if split_set in ["train", "train-clip", "val", "val-clip"]:
            self.data_path = os.path.join(root_dir, "training")
            velodyne_files = sorted(os.listdir(os.path.join(self.data_path, 'velodyne')))
            all_ids = [f[:-len('.txt')] for f in velodyne_files]
            all_labels = [f[:-len('.txt')] for f in velodyne_files]


            x_train, x_validate, y_train, y_validate = train_test_split(all_ids, all_labels, test_size=0.25, random_state=612932)
            self.ids = x_train if split_set in ['train', 'train-clip'] else x_validate
            self.labels = y_train if split_set in ['train', 'train-clip'] else y_validate

            self.data_len = len(self.ids)
        else:
            self.data_path_video = os.path.join(root_dir, 'raw/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data')
            self.data_path_video_calib = os.path.join(root_dir, 'raw/2011_09_26')
            self.data_len = len(os.listdir(self.data_path_video))

        self.num_points = num_points
        self.augment = augment
        self.use_height = use_height
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(
            min_points=random_cuboid_min_points,
            aspect=0.75,
            min_crop=0.75,
            max_crop=1.0,
        )
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = 64

    def __len__(self):
        return self.data_len

    def _process_point_cloud(self, point_cloud, calib):
        # Only capture the part of the point cloud that is visible from the camera
        point_cloud_proj, point_cloud_proj_infront = calib.project_velo_to_image(point_cloud)
        visible_mask = \
            (point_cloud_proj[:, 0] >= 0) & \
            (point_cloud_proj[:, 1] >= 0) & \
            (point_cloud_proj[:, 0] < 1224) & \
            (point_cloud_proj[:, 1] < 370) & \
            (point_cloud_proj_infront)
        point_cloud = point_cloud[visible_mask]

        return pc_util.random_sampling(point_cloud, self.num_points)

    def __getitem__(self, idx):

        point_cloud = None
        bboxes = None
        point_cloud_video = None

        number_id = idx

        if self.split_set in ["train", "train-clip", "val", "val-clip"]:
            string_id, number_id = self.ids[idx], int(self.ids[idx])
            label_id = self.labels[idx]
            point_cloud = load_velo_scan(os.path.join(self.data_path, 'velodyne', f'{string_id}.bin'))[:, 0:3]
            calib = Calibration(os.path.join(self.data_path, 'calib', f'{string_id}.txt'))
            objects = read_label(os.path.join(self.data_path, 'label_2', f'{label_id}.txt'))
            # Only use objects of classes with enough data
            objects = [object for object in objects if object.type in self.dataset_config.type2class.keys()]

            bboxes = []
            for object in objects:
                box3d_pts_3d = compute_box_3d(object, calib.P)[1]
                box3d_pts_3d = calib.project_rect_to_velo(box3d_pts_3d)
                obb = convert_box_corners_into_obb(box3d_pts_3d)
                bboxes.append(np.array([
                    obb[0], obb[1], obb[2],
                    obb[3], obb[4], obb[5],
                    obb[6],
                    self.dataset_config.type2class[object.type],
                ]))
            bboxes = np.array(bboxes)

            point_cloud = self._process_point_cloud(point_cloud, calib)

            point_cloud_video = np.array([[[]]], dtype=np.float32)
            if self.split_set in ["train-clip", "val-clip"]:
                calib_video = self.raw_mapper.load_calibration_from_video(number_id)
                point_cloud_video = self.raw_mapper.load_previous_velo_video_from_compressed(number_id, clip_size=4)
                point_cloud_video = np.stack([self._process_point_cloud(point_cloud_frame, calib_video) for point_cloud_frame in point_cloud_video])
        else:
            #TODO refactor imp details of kitti out of here and to raw_mapper
            calib_video = self.raw_mapper.load_calibration_from_video_path(self.data_path_video_calib)
            point_cloud = self._process_point_cloud(
                self.raw_mapper.load_velo(f'{self.data_path_video}/{number_id:010d}.bin.npz'),
                calib_video,
            )
            bboxes = np.zeros((0, 8))
            point_cloud_video = np.array([[[]]], dtype=np.float32)


        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = np.pi - bboxes[:, 6]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = pc_util.rotz(rot_angle)

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 6] -= rot_angle

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio
            bboxes[:, 0:3] *= scale_ratio
            bboxes[:, 3:6] *= scale_ratio

            if self.use_height:
                point_cloud[:, -1] *= scale_ratio[0, 0]

            if self.use_random_cuboid:
                point_cloud, bboxes, _ = self.random_cuboid_augmentor(
                    point_cloud, bboxes
                )

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

        point_cloud_dims_min = point_cloud.min(axis=0)
        point_cloud_dims_max = point_cloud.max(axis=0)

        #TODO code duplication
        point_cloud_video_dims_min = np.array([], dtype=np.float32)
        point_cloud_video_dims_max = np.array([], dtype=np.float32)
        if self.split_set in ["train-clip", "val-clip"]:
            point_cloud_video_dims_min = np.stack([point_cloud_frame.min(axis=0) for point_cloud_frame in point_cloud_video])
            point_cloud_video_dims_max = np.stack([point_cloud_frame.max(axis=0) for point_cloud_frame in point_cloud_video])

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

        #NOTE This translates the corners out of the space they are currently in (out of velo axis orientation?)
        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["point_cloud_prev_clips"] = point_cloud_video.astype(np.float32)
        ret_dict["point_cloud_prev_clips_dims_min"] = point_cloud_video_dims_min
        ret_dict["point_cloud_prev_clips_dims_max"] = point_cloud_video_dims_max
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        target_bboxes_semcls = np.zeros((self.max_num_obj))
        target_bboxes_semcls[0 : bboxes.shape[0]] = bboxes[:, -1]
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(number_id).astype(np.int64)
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes
        ret_dict["gt_angle_residual_label"] = angle_residuals
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max

        return ret_dict
