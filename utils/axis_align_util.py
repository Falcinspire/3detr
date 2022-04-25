from datasets.kitti_util import roty
from utils.pc_util import rotz
import numpy as np

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

def convert_box_corners_into_obb(box3d_pts_3d):
    box3d_pts_3d = box3d_pts_3d.copy()
    
    v2 = box3d_pts_3d[1][:2]
    v1 = box3d_pts_3d[0][:2]
    angle = angle_between(v2 - v1, np.array([1, 0]))
    # cross product is positive if v2 is closer clockwise than counterclockwise from v1. I'm pretty sure anyways.
    if np.cross(v1, v2) > 0: angle = -angle

    label_center = np.mean(box3d_pts_3d, axis=0)
    box3d_pts_3d -= label_center
    box3d_pts_3d = np.matmul(box3d_pts_3d, rotz(-angle))
    box3d_pts_3d += label_center

    new_bbox = np.array([
        np.min(box3d_pts_3d[:,0]), np.min(box3d_pts_3d[:,1]), np.min(box3d_pts_3d[:,2]), 
        np.max(box3d_pts_3d[:,0]), np.max(box3d_pts_3d[:,1]), np.max(box3d_pts_3d[:,2]),
    ])

    width, length, height = (new_bbox[3] - new_bbox[0]) / 2, (new_bbox[4] - new_bbox[1]) / 2, (new_bbox[5] - new_bbox[2]) / 2
    return np.array([
        new_bbox[0] + width, new_bbox[1] + length, new_bbox[2] + height,
        width, length, height,
        angle,
    ])

def convert_box_corners_into_obb_y_upward(box3d_pts_3d):
    box3d_pts_3d = box3d_pts_3d.copy()
    
    v2 = box3d_pts_3d[1][[0, 2]]
    v1 = box3d_pts_3d[0][[0, 2]]
    angle = angle_between(v2 - v1, np.array([1, 0]))
    # cross product is positive if v2 is closer clockwise than counterclockwise from v1. I'm pretty sure anyways.
    if np.cross(v1, v2) > 0: angle = -angle

    label_center = np.mean(box3d_pts_3d, axis=0)
    box3d_pts_3d -= label_center
    box3d_pts_3d = np.matmul(box3d_pts_3d, roty(-angle))
    box3d_pts_3d += label_center

    new_bbox = np.array([
        np.min(box3d_pts_3d[:,0]), np.min(box3d_pts_3d[:,1]), np.min(box3d_pts_3d[:,2]), 
        np.max(box3d_pts_3d[:,0]), np.max(box3d_pts_3d[:,1]), np.max(box3d_pts_3d[:,2]),
    ])

    width, length, height = (new_bbox[3] - new_bbox[0]) / 2, (new_bbox[4] - new_bbox[1]) / 2, (new_bbox[5] - new_bbox[2]) / 2
    return np.array([
        new_bbox[0] + width, new_bbox[1] + length, new_bbox[2] + height,
        width, length, height,
        angle,
    ])
