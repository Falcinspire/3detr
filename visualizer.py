from datasets.kitti import KITTI3DObjectDetectionDataset, KITTI3DObjectDetectionDatasetConfig
from datasets.kitti_util import Calibration, load_velo_scan, read_label
from datasets.sunrgbd import SunrgbdDatasetConfig, SunrgbdDetectionDataset
import open3d as o3d
import numpy as np
import utils.pc_util as pc_util
import os
import torch
import random
import matplotlib.pyplot as plt

#refs https://stackoverflow.com/a/66442894
def draw_box(corners, color):
    # corners = [[corner[0], corner[2], corner[1]] for corner in corners]

    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)

    return line_set

def draw_boxes(boxes, color):
    return [draw_box(box, color) for box in boxes]

def draw_spheres(locs, size=0.2):
    return [o3d.geometry.TriangleMesh.create_sphere(size).translate(loc) for loc in locs]

# def render_image():
#     img_width, img_height = 1920, 1080
#     renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
#     renderer_pc.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))

#     mtl = o3d.visualization.rendering.MaterialRecord()
#     mtl.base_color = [0.0, 0.0, 0.0, 1.0]  # RGBA
#     mtl.shader = "defaultUnlit"

#     renderer_pc.scene.add_geometry("pcd", pcd, mtl)

#     # Optionally set the camera field of view (to zoom in a bit)
#     vertical_field_of_view = 60.0  # between 5 and 90 degrees
#     aspect_ratio = img_width / img_height  # azimuth over elevation
#     near_plane = 0.1
#     far_plane = 1000.0
#     fov_type = o3d.visualization.rendering.Camera.FovType.Horizontal
#     renderer_pc.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

#     # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
#     center = [40, 0, -10]  # look_at target
#     eye = [-10, 0, 10]  # camera position
#     up = [0, 0, 1]  # camera orientation
#     # center = [cloud[:, :0].mean(), cloud[:, :1].mean(), cloud[:, :2].mean()]  # look_at target
#     # eye = [0, 0, 10]  # camera position
#     # up = [0, 0, 1]  # camera orientation
#     renderer_pc.scene.camera.look_at(center, eye, up)

#     image = renderer_pc.render_to_image()
#     o3d.io.write_image(f"./screenshot_{str(i)}.png", image, 9)

#TODO DO NOT COMMIT
#TODO DO NOT COMMIT
#TODO DO NOT COMMIT
#TODO DO NOT COMMIT
#TODO DO NOT COMMIT
#TODO DO NOT COMMIT

# ds = KITTI3DObjectDetectionDataset(
#     KITTI3DObjectDetectionDatasetConfig(), 
#     root_dir='D:\MLDataset\KITTI_ObjectDetection',
#     augment=True,
# )
# for value in ds:
#     point_cloud = value['point_clouds']
#     labels = value['velo_gt_box_corners']
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(point_cloud)
#     bboxes = draw_boxes(labels)
#     o3d.visualization.draw_geometries([pcd] + bboxes)

# files = os.listdir('D:/MLDataset/KITTI/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/')
# random.shuffle(files)
# for file in files:
#     scan = np.fromfile(f'D:/MLDataset/KITTI/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/{file}', dtype=np.float32)
#     scan = scan.reshape((-1, 4))[:, :3]
#     scan = pc_util.random_sampling(scan, 400000)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(scan)

#     sphere = o3d.geometry.TriangleMesh.create_sphere(0.2).translate((0, 10, 0))

#     o3d.visualization.draw_geometries([pcd] + draw_spheres([(0, 0, 10), (1, 1, 10)]))



#TODO refactor this out
def _process_point_cloud(point_cloud, calib):
    # Only capture the part of the point cloud that is visible from the camera
    point_cloud_proj, point_cloud_proj_infront = calib.project_velo_to_image(point_cloud)
    visible_mask = \
        (point_cloud_proj[:, 0] >= 0) & \
        (point_cloud_proj[:, 1] >= 0) & \
        (point_cloud_proj[:, 0] < 1224) & \
        (point_cloud_proj[:, 1] < 370) & \
        (point_cloud_proj_infront)
    point_cloud = point_cloud[visible_mask]

    return pc_util.random_sampling(point_cloud, 20000)

def get_point_color(point):
    return [0.0, 1.0, 0.0];

# img_width, img_height = 1920, 1080
# renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
# renderer_pc.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))

# for i in range(100):
#     # calib = Calibration(f'D:/MLDataset/KITTI_ObjectDetection/training/calib/{i:06d}.txt')
#     calib = Calibration('D:/MLDataset/KITTI_ObjectDetection/2011_09_26', from_video=True)
#     point_cloud = _process_point_cloud(load_velo_scan(f'D:\MLDataset\KITTI/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/{i:010d}.bin')[:, 0:3], calib)
#     # objects = read_label(f'D:/MLDataset/KITTI_ObjectDetection/training/label_2/{i:06d}.txt')
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(point_cloud)
#     pcd.colors = o3d.utility.Vector3dVector([get_point_color(point) for point in point_cloud])


#     mtl = o3d.visualization.rendering.MaterialRecord()
#     mtl.base_color = [0.0, 0.0, 0.0, 1.0]  # RGBA
#     mtl.shader = "defaultUnlit"

#     renderer_pc.scene.add_geometry('pcd', pcd, mtl)

#     # Optionally set the camera field of view (to zoom in a bit)
#     vertical_field_of_view = 60.0  # between 5 and 90 degrees
#     aspect_ratio = img_width / img_height  # azimuth over elevation
#     near_plane = 0.1
#     far_plane = 1000.0
#     fov_type = o3d.visualization.rendering.Camera.FovType.Horizontal
#     renderer_pc.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

#     # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
#     center = [40, 0, -10]  # look_at target
#     eye = [-10, 0, 10]  # camera position
#     up = [0, 0, 1]  # camera orientation
#     # center = [cloud[:, :0].mean(), cloud[:, :1].mean(), cloud[:, :2].mean()]  # look_at target
#     # eye = [0, 0, 10]  # camera position
#     # up = [0, 0, 1]  # camera orientation
#     renderer_pc.scene.camera.look_at(center, eye, up)

#     image = renderer_pc.render_to_image()
#     o3d.io.write_image(f"./screenshot_{i}.png", image, 9)

#     renderer_pc.scene.remove_geometry('pcd')

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# center = [0, 0, 0]  # look_at target
# eye = [0.0, 0.0, 100.0]  # camera position
# up = [0, 0, 1]  # camera orientation
# vis.get_view_control().camera_local_translate(*eye)
# vis.get_view_control().set_lookat(center)
# vis.get_view_control().set_up(up)
# vis.add_geometry(pcd)
# vis.update_geometry(pcd)
# vis.poll_events()
# vis.update_renderer()
# vis.capture_screen_image(f'screenshot_101.png', do_render=True)
# vis.destroy_window()









img_width, img_height = 1920, 1080
renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
renderer_pc.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))

counter = 0
for i in range(4):
    value = torch.load(f'visualizations/0_{i}.pt')
    # point_cloud_batches = value['batch_data_label']['point_clouds']
    point_cloud_batches = value['inputs']['point_clouds']
    query_point_batches = value['query_points']
    incoming_detection_batches = value['incoming_detections']
    # box_corner_batches = value['outputs']['outputs']['velo_box_corners']
    # label_batches = value['batch_data_label']['velo_gt_box_corners']
    # box_corner_batches = value['outputs']['outputs']['velo_box_corners']
    predicted_probabilities_batches = value['outputs']['outputs']['objectness_prob']
    conf_thresh = 0.05

    for point_cloud, query_points, incoming_detections, predicted_probabilities in zip(point_cloud_batches, query_point_batches, incoming_detection_batches, predicted_probabilities_batches):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector([get_point_color(point) for point in point_cloud])

        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [0.0, 0.0, 0.0, 1.0]  # RGBA
        mtl.shader = "defaultUnlit"

        mtl2 = o3d.visualization.rendering.MaterialRecord()
        mtl2.base_color = [0.0, 1.0, 0.0, 1.0]  # RGBA
        mtl2.shader = "defaultUnlit"

        mtl3 = o3d.visualization.rendering.MaterialRecord()
        mtl3.base_color = [0.0, 0.0, 1.0, 1.0]  # RGBA
        mtl3.shader = "defaultUnlit"

        renderer_pc.scene.add_geometry('pcd', pcd, mtl)
        for q_idx, q in enumerate(draw_spheres(query_points.cpu().numpy())):
            renderer_pc.scene.add_geometry(f'qp{q_idx}', q, mtl2)
        for id_idx, i in enumerate(draw_spheres(incoming_detections.cpu().numpy(), size=0.3)):
            renderer_pc.scene.add_geometry(f'id{id_idx}', i, mtl3)

        # Optionally set the camera field of view (to zoom in a bit)
        vertical_field_of_view = 60.0  # between 5 and 90 degrees
        aspect_ratio = img_width / img_height  # azimuth over elevation
        near_plane = 0.1
        far_plane = 1000.0
        fov_type = o3d.visualization.rendering.Camera.FovType.Horizontal
        renderer_pc.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

        # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
        center = [40, 0, -10]  # look_at target
        eye = [-10, 0, 10]  # camera position
        up = [0, 0, 1]  # camera orientation
        # center = [cloud[:, :0].mean(), cloud[:, :1].mean(), cloud[:, :2].mean()]  # look_at target
        # eye = [0, 0, 10]  # camera position
        # up = [0, 0, 1]  # camera orientation
        renderer_pc.scene.camera.look_at(center, eye, up)

        image = renderer_pc.render_to_image()
        o3d.io.write_image(f"./screenshot_{counter}.png", image, 9)
        counter += 1

        renderer_pc.scene.remove_geometry('pcd')
        for q_idx in range(len(query_points)):
            renderer_pc.scene.remove_geometry(f'qp{q_idx}')
        for id_idx in range(len(incoming_detections)):
            renderer_pc.scene.remove_geometry(f'id{id_idx}')













# for i in range(4):
#     value = torch.load(f'visualizations/0_{i}.pt')
#     # point_cloud_batches = value['batch_data_label']['point_clouds']
#     point_cloud_batches = value['inputs']['point_clouds']
#     query_point_batches = value['query_points']
#     # label_batches = value['batch_data_label']['velo_gt_box_corners']
#     # box_corner_batches = value['outputs']['outputs']['velo_box_corners']
#     predicted_probabilities_batches = value['outputs']['outputs']['objectness_prob']
#     conf_thresh = 0.05
#     # for point_cloud, query_points, labels, box_corners, predicted_probabilities in zip(point_cloud_batches, query_point_batches, label_batches, box_corner_batches, predicted_probabilities_batches):
#     for point_cloud, query_points, predicted_probabilities in zip(point_cloud_batches, query_point_batches, predicted_probabilities_batches):
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())

#         spheres = draw_spheres(query_points.cpu())

#         # bboxes = draw_boxes(labels.cpu(), [1.0, 0.0, 0.0])

#         # predicted_bboxes = draw_boxes(box_corners[predicted_probabilities >= conf_thresh].cpu(), [0.0, 1.0, 0.0])

#         # o3d.visualization.draw_geometries([pcd] + spheres + bboxes + predicted_bboxes)

#         o3d.visualization.draw_geometries([pcd] + spheres)





# ds = KITTI3DObjectDetectionDataset(
#     KITTI3DObjectDetectionDatasetConfig(), 
#     root_dir='D:\MLDataset\KITTI_ObjectDetection',
# )
# for item in ds:
#     pass


# item = ds[10]
# point_cloud = item['point_clouds']
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud)
# o3d.visualization.draw_geometries([pcd] + draw_boxes(item['velo_gt_box_corners']))

# ds = KITTI3DObjectDetectionDataset(
#     KITTI3DObjectDetectionDatasetConfig(), 
#     root_dir='D:\MLDataset\KITTI_ObjectDetection',
#     use_clip_of_size=4,
# )
# item = ds[20]
# point_cloud_prev = item['point_cloud_prev_clips']
# point_cloud = item['point_clouds']
# full_point_cloud = np.concatenate([point_cloud_prev, point_cloud[None, :, :]])
# for i, cloud in enumerate(full_point_cloud):
# # labels = [[[corner[0], corner[2], corner[1]] for corner in corners] for corners in labels]
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(cloud)
#     o3d.visualization.draw_geometries([pcd])

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.update_geometry(pcd)
    # vis.poll_events()
    # vis.update_renderer()
    # center = [40, 0, -10]  # look_at target
    # eye = [-10, 0, 10]  # camera position
    # up = [0, 0, 1]  # camera orientation
    # vis.get_view_control().set_lookat(center)
    # vis.get_view_control().set_up(up)
    # vis.capture_screen_image(f'screenshot_{str(i)}.png', do_render=True)
    # vis.destroy_window()

    

# for i in range(1):
#     value = torch.load(f'{str(i)}.pt')
#     point_cloud = value['batch_data_label']['point_clouds'][0][:10].cpu().numpy()
#     print(point_cloud.shape)
#     point_cloud = np.stack([point_cloud[:, 0], point_cloud[:, 2], point_cloud[:, 1]], axis=-1)
#     print(point_cloud.shape)
















# ds = KITTI3DObjectDetectionDataset(
#     KITTI3DObjectDetectionDatasetConfig(), 
#     root_dir='D:\MLDataset\KITTI_ObjectDetection',
#     split_set="video"
#     # use_clip_of_size=4,
# )
# # for i in range(0, len(ds)):
# #     try:
# #         item = ds[i]
# #         break
# #     except:
# #         print(f'oof {i}')
# # point_cloud_prev = item['point_cloud_prev_clips']
# # point_cloud = item['point_clouds']
# # full_point_cloud = np.array([point_cloud])
# # full_point_cloud = np.concatenate([point_cloud_prev, point_cloud[None, :, :]])
# # full_point_cloud = [ds[i]['point_clouds'] for i in range(2)]

# img_width, img_height = 1920, 1080
# renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
# renderer_pc.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))
# # for i, point_cloud in enumerate(full_point_cloud):
# for i in range(20):
#     point_cloud = ds[i]['point_clouds']
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(point_cloud)
#     pcd.colors = o3d.utility.Vector3dVector([get_point_color(point) for point in point_cloud])

#     mtl = o3d.visualization.rendering.MaterialRecord()
#     mtl.base_color = [0.0, 0.0, 0.0, 1.0]  # RGBA
#     mtl.shader = "defaultUnlit"

#     renderer_pc.scene.add_geometry('pcd', pcd, mtl)

#     # Optionally set the camera field of view (to zoom in a bit)
#     vertical_field_of_view = 60.0  # between 5 and 90 degrees
#     aspect_ratio = img_width / img_height  # azimuth over elevation
#     near_plane = 0.1
#     far_plane = 1000.0
#     fov_type = o3d.visualization.rendering.Camera.FovType.Horizontal
#     renderer_pc.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

#     # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
#     center = [40, 0, -10]  # look_at target
#     eye = [-10, 0, 10]  # camera position
#     up = [0, 0, 1]  # camera orientation
#     # center = [cloud[:, :0].mean(), cloud[:, :1].mean(), cloud[:, :2].mean()]  # look_at target
#     # eye = [0, 0, 10]  # camera position
#     # up = [0, 0, 1]  # camera orientation
#     renderer_pc.scene.camera.look_at(center, eye, up)

#     image = renderer_pc.render_to_image()
#     o3d.io.write_image(f"./screenshot_{i}.png", image, 9)

#     renderer_pc.scene.remove_geometry('pcd')