# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import datetime
import logging
import math
import time
import sys
import itertools
import os
from os import path
import numpy as np

from torch.distributed.distributed_c10d import reduce
from datasets.kitti_util import Calibration
from utils.ap_calculator import APCalculator, flip_axis_to_depth, get_ap_config_dict, parse_predictions
from utils.axis_align_util import convert_box_corners_into_obb, convert_box_corners_into_obb_y_upward
from utils.box_util import flip_axis_to_camera_np
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
)
from utils.open3d_renderer_util import Open3dInteractiveRendererUtil, Open3dOfflineRendererUtil


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
):

    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=False,
    )

    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        outputs = model(inputs)

        # Compute loss
        loss, loss_dict = criterion(outputs, batch_data_label)

        loss_reduced = all_reduce_average(loss)
        loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)

        loss.backward()
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        if curr_iter % args.log_metrics_every == 0:
            # This step is slow. AP is computed approximately and locally during training.
            # It will gather outputs and ground truth across all ranks.
            # It is memory intensive as point_cloud ground truth is a large tensor.
            # If GPU memory is not an issue, uncomment the following lines.
            # outputs["outputs"] = all_gather_dict(outputs["outputs"])
            # batch_data_label = all_gather_dict(batch_data_label)
            ap_calculator.step_meter(outputs, batch_data_label)

        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict["batch_time"] = time_delta.avg
            logger.log_scalars(train_dict, curr_iter, prefix="Train/")

        curr_iter += 1
        barrier()

    return ap_calculator


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
):

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        outputs = model(inputs)

        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)
        ap_calculator.step_meter(outputs, batch_data_label)
        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg
        curr_iter += 1
        barrier()
    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

    return ap_calculator

#TODO this reaks of code duplication
@torch.no_grad()
def predict_only(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
    predict_with_query_reuse,
):
    #TODO allow prediction of multiple batches, also refactor
    if not path.isdir(args.predict_output):
        os.makedirs(args.predict_output)

    net_device = next(model.parameters()).device

    ap_config_dict = get_ap_config_dict(
        dataset_config=dataset_config, remove_empty_box=True,
        conf_thresh=0.05,
    )

    model.eval()
    barrier()

    num_reused_query_points = []
    num_reused_query_points_used = []

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()

        batch_cnt = len(batch_data_label['point_clouds'])

        batch_videos = [
            {
                "point_clouds_video": [clip for clip in point_cloud_prev_clips] + [point_clouds],
                "point_cloud_dims_min_video": [clip for clip in point_cloud_prev_clips_dims_min] + [point_cloud_dims_min],
                "point_cloud_dims_max_video": [clip for clip in point_cloud_prev_clips_dims_max] + [point_cloud_dims_max],
            }
            for (
                point_cloud_prev_clips, 
                point_clouds, 
                point_cloud_prev_clips_dims_min, 
                point_cloud_prev_clips_dims_max, 
                point_cloud_dims_min, 
                point_cloud_dims_max
            ) in zip(
                batch_data_label['point_cloud_prev_clips'], 
                batch_data_label['point_clouds'],
                batch_data_label['point_cloud_prev_clips_dims_min'],
                batch_data_label['point_cloud_prev_clips_dims_max'],
                batch_data_label['point_cloud_dims_min'],
                batch_data_label['point_cloud_dims_max'],
            )
        ]

        input_repeated = [
            {
                "point_clouds": torch.stack([batch_videos[j]["point_clouds_video"][i] for j in range(batch_cnt)]),
                "point_cloud_dims_min": torch.stack([batch_videos[j]["point_cloud_dims_min_video"][i] for j in range(batch_cnt)]),
                "point_cloud_dims_max": torch.stack([batch_videos[j]["point_cloud_dims_max_video"][i] for j in range(batch_cnt)]),
            }
            for i in range(len(batch_videos[0]['point_clouds_video']))
        ]

        prev_detections = torch.zeros((batch_cnt, 0, 3))
        for local_idx, inputs in enumerate(input_repeated):
            # Do not run first 3 clips if query reuse is toggled off
            if (not predict_with_query_reuse) and (local_idx < len(input_repeated)-1): continue

            for key in inputs:
                inputs[key] = inputs[key].to(net_device)

            outputs, query_xyz = model(inputs, return_queries=True, prev_detections=prev_detections)

            # Memory intensive as it gathers point cloud GT tensor across all ranks
            outputs["outputs"] = all_gather_dict(outputs["outputs"])
            inputs = all_gather_dict(inputs)

            predicted_box_corners=outputs['outputs']["box_corners"]
            sem_cls_probs=outputs['outputs']["sem_cls_prob"]
            pred_sem_cls = torch.argmax(sem_cls_probs, dim=-1)
            objectness_probs=outputs['outputs']["objectness_prob"]
            center_unnormalized=outputs['outputs']['center_unnormalized']
            point_cloud=batch_data_label["point_clouds"]

            for idx, (center_unnormalized_each, pred_sem_cls_each, objectness_probs_each) in enumerate(zip(center_unnormalized, pred_sem_cls, objectness_probs)):
                queries = (pred_sem_cls_each < dataset_config.num_semcls) & (objectness_probs_each > 0.05) #TODO magic number
                if len(num_reused_query_points) == 0:
                    num_reused_query_points = [0 for _ in batch_data_label['point_clouds']]
                    num_reused_query_points_used = [0 for _ in batch_data_label['point_clouds']]
                num_reused_query_points[idx] += len(prev_detections)
                num_reused_query_points_used[idx] += queries[:len(prev_detections)].sum().item()

            print([a/b for a, b in zip(num_reused_query_points_used, num_reused_query_points)])

            batches = []
            for center_unnormalized_each, pred_sem_cls_each, objectness_probs_each in zip(center_unnormalized, pred_sem_cls, objectness_probs):
                queries = center_unnormalized_each[(pred_sem_cls_each < dataset_config.num_semcls) & (objectness_probs_each > 0.05)] #TODO magic number
                queries = queries.cpu()
                batches.append(queries)
            prev_detections = batches

            batch_pred_map_cls = parse_predictions(
                predicted_box_corners,
                sem_cls_probs,
                objectness_probs,
                point_cloud,
                ap_config_dict,
            )

            if local_idx == len(input_repeated) - 1:
                #TODO don't use args value for dataset here. Maybe include in kitti getitem()?
                calib = Calibration(os.path.join(args.dataset_root_dir, 'training', 'calib', f'{batch_data_label["scan_idx"][0].item():06d}.txt'))

                batch_we_care_about_for_now = batch_pred_map_cls[0]

                filepath = path.join(args.predict_output, f'{(batch_data_label["scan_idx"][0].item()):06d}.txt')
                print(filepath)
                with open(filepath, 'w+') as out:
                    for predicted_object in batch_we_care_about_for_now:
                        sem_class, box_corners, confidence = predicted_object
                        box_corners = flip_axis_to_depth(box_corners) # Back to velo space we go
                        obb = convert_box_corners_into_obb_y_upward(calib.project_velo_to_rect(box_corners)) #TODO this should be consistent with what happens in kitti.py

                        image_coords = np.array(calib.project_velo_to_image(box_corners)[0])
                        xmin = max(0, np.min(image_coords[:, 0])) #TODO constrain these to image coords
                        ymin = max(0, np.min(image_coords[:, 1]))
                        xmax = min(1223, np.max(image_coords[:, 0]))
                        ymax = min(369, np.max(image_coords[:, 1]))

                        type = dataset_config.class2type[sem_class]
                        truncated = -1
                        occluded = -1
                        alpha = -10

                        #TODO This entire section was developed through guessing and trial and error
                        # Benchmark uses bottom-center of box. In camera coords, -y is up.
                        coords_3d_pos = [obb[0], obb[1] + obb[4], obb[2]]
                        # height, width, length. *2 because obb func uses half-sizes
                        coords_3d_dimensions = obb[[4,5,3]]*2.0
                        # Must be negative because negative y is up
                        coords_3d_ry = -obb[6]
                        coords_2d = [xmin, ymax, xmax, ymin]
                        score = confidence

                        out.write(
                            f'{type} {truncated} {occluded} {alpha} {coords_2d[0]:.2f} {coords_2d[1]:.2f} {coords_2d[2]:.2f} {coords_2d[3]:.2f} {coords_3d_dimensions[0]:.2f} {coords_3d_dimensions[1]:.2f} {coords_3d_dimensions[2]:.2f} {coords_3d_pos[0]:.2f} {coords_3d_pos[1]:.2f} {coords_3d_pos[2]:.2f} {coords_3d_ry:.2f} {score:.2f}\n'
                        )

    barrier()

@torch.no_grad()
def render_only(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
):
    #TODO allow rendering of multiple batches, also refactor
    if not path.isdir(args.render_output):
        os.makedirs(args.render_output)

    net_device = next(model.parameters()).device

    ap_config_dict = get_ap_config_dict(
        dataset_config=dataset_config, remove_empty_box=True,
        conf_thresh=0.05,
    )

    renderer = Open3dOfflineRendererUtil(1920, 1080)

    model.eval()
    barrier()

    if (args.render_kitti_dataset == 'kitti-clip'):
        for batch_idx, batch_data_label in enumerate(dataset_loader):
            curr_time = time.time()

            batch_cnt = len(batch_data_label['point_clouds'])

            batch_videos = [
                {
                    "point_clouds_video": [clip for clip in point_cloud_prev_clips] + [point_clouds],
                    "point_cloud_dims_min_video": [clip for clip in point_cloud_prev_clips_dims_min] + [point_cloud_dims_min],
                    "point_cloud_dims_max_video": [clip for clip in point_cloud_prev_clips_dims_max] + [point_cloud_dims_max],
                }
                for (
                    point_cloud_prev_clips, 
                    point_clouds, 
                    point_cloud_prev_clips_dims_min, 
                    point_cloud_prev_clips_dims_max, 
                    point_cloud_dims_min, 
                    point_cloud_dims_max
                ) in zip(
                    batch_data_label['point_cloud_prev_clips'], 
                    batch_data_label['point_clouds'],
                    batch_data_label['point_cloud_prev_clips_dims_min'],
                    batch_data_label['point_cloud_prev_clips_dims_max'],
                    batch_data_label['point_cloud_dims_min'],
                    batch_data_label['point_cloud_dims_max'],
                )
            ]

            input_repeated = [
                {
                    "point_clouds": torch.stack([batch_videos[j]["point_clouds_video"][i] for j in range(batch_cnt)]),
                    "point_cloud_dims_min": torch.stack([batch_videos[j]["point_cloud_dims_min_video"][i] for j in range(batch_cnt)]),
                    "point_cloud_dims_max": torch.stack([batch_videos[j]["point_cloud_dims_max_video"][i] for j in range(batch_cnt)]),
                }
                for i in range(len(batch_videos[0]['point_clouds_video']))
            ]

            prev_detections = torch.zeros((batch_cnt, 0, 3))
            last_prev_detections = torch.zeros((batch_cnt, 0, 3))
            for local_idx, inputs in enumerate(input_repeated):
                for key in inputs:
                    inputs[key] = inputs[key].to(net_device)

                outputs, query_xyz = model(inputs, return_queries=True, prev_detections=prev_detections)

                # Memory intensive as it gathers point cloud GT tensor across all ranks
                outputs["outputs"] = all_gather_dict(outputs["outputs"])
                inputs = all_gather_dict(inputs)

                predicted_box_corners=outputs['outputs']["box_corners"]
                sem_cls_probs=outputs['outputs']["sem_cls_prob"]
                pred_sem_cls = torch.argmax(sem_cls_probs, dim=-1)
                objectness_probs=outputs['outputs']["objectness_prob"]
                center_unnormalized=outputs['outputs']['center_unnormalized']
                point_cloud=batch_data_label["point_clouds"]

                batches = []
                for center_unnormalized_each, pred_sem_cls_each, objectness_probs_each in zip(center_unnormalized, pred_sem_cls, objectness_probs):
                    queries = center_unnormalized_each[(pred_sem_cls_each < dataset_config.num_semcls) & (objectness_probs_each > 0.05)] #TODO magic number
                    queries = queries.cpu()
                    batches.append(queries)
                last_prev_detections = [detections.numpy() for detections in prev_detections]
                prev_detections = batches

                batch_pred_map_cls = parse_predictions(
                    predicted_box_corners,
                    sem_cls_probs,
                    objectness_probs,
                    point_cloud,
                    ap_config_dict,
                )

                point_cloud = point_cloud.cpu().detach().numpy()
                query_xyz = query_xyz.cpu().detach().numpy()

                point_cloud = flip_axis_to_camera_np(point_cloud)
                last_prev_detections = [flip_axis_to_camera_np(detections) for detections in last_prev_detections]
                query_xyz = np.stack([flip_axis_to_camera_np(queries) for queries in query_xyz])

                if local_idx < len(input_repeated) - 1:
                    renderer.draw_point_cloud(point_cloud[0])
                    for query in query_xyz[0]:
                        renderer.draw_sphere(query, color=[0.2, 0.4, 0.2])
                    for detection in last_prev_detections[0]:
                        renderer.draw_sphere(detection, size=0.3, color=[1.0, 0.0, 0.0])
                    for box in batch_pred_map_cls[0]:
                        renderer.draw_box(box[1])
                    filepath = path.join(args.render_output, f'{batch_idx}_{local_idx}.png')
                    print(filepath)
                    renderer.render_image(filepath)
                    renderer.clear_scene()
                else:
                    gt_box_corners=batch_data_label["gt_box_corners"]
                    gt_box_sem_cls_labels=batch_data_label["gt_box_sem_cls_label"]
                    gt_box_present=batch_data_label["gt_box_present"]

                    #NOTE This code is copied from ap_calculator.step()
                    gt_box_corners = gt_box_corners.cpu().detach().numpy()
                    gt_box_sem_cls_labels = gt_box_sem_cls_labels.cpu().detach().numpy()
                    gt_box_real = gt_box_sem_cls_labels != dataset_config.num_semcls
                    gt_box_present = gt_box_present.cpu().detach().numpy()

                    renderer.draw_point_cloud(point_cloud[0])
                    for gt_box in gt_box_corners[0][gt_box_real[0]]:
                        renderer.draw_box(gt_box, color=[0.0, 0.5, 0.0])
                    for query in query_xyz[0]:
                        renderer.draw_sphere(query, color=[0.2, 0.4, 0.2])
                    for detection in last_prev_detections[0]:
                        renderer.draw_sphere(detection, size=0.3, color=[1.0, 0.0, 0.0])
                    for box in batch_pred_map_cls[0]:
                        renderer.draw_box(box[1])
                    filepath = path.join(args.render_output, f'{batch_idx}_{local_idx}.png')
                    print(filepath)
                    renderer.render_image(filepath)
                    renderer.clear_scene()
    elif (args.render_kitti_dataset == 'kitti-video'):
        prev_detections = None
        last_prev_detections = None

        for batch_idx, batch_data_label in enumerate(dataset_loader):
            curr_time = time.time()

            batch_cnt = len(batch_data_label['point_clouds'])

            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(net_device)

            if prev_detections == None:
                prev_detections = torch.zeros((batch_cnt, 0, 3))
                last_prev_detections = torch.zeros((batch_cnt, 0, 3))

            inputs = {
                "point_clouds": batch_data_label["point_clouds"],
                "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
                "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
            }
            outputs, query_xyz = model(inputs, return_queries=True, prev_detections=prev_detections)

            # Memory intensive as it gathers point cloud GT tensor across all ranks
            outputs["outputs"] = all_gather_dict(outputs["outputs"])
            inputs = all_gather_dict(inputs)

            predicted_box_corners=outputs['outputs']["box_corners"]
            sem_cls_probs=outputs['outputs']["sem_cls_prob"]
            pred_sem_cls = torch.argmax(sem_cls_probs, dim=-1)
            objectness_probs=outputs['outputs']["objectness_prob"]
            center_unnormalized=outputs['outputs']['center_unnormalized']
            point_cloud=batch_data_label["point_clouds"]

            batches = []
            for center_unnormalized_each, pred_sem_cls_each, objectness_probs_each in zip(center_unnormalized, pred_sem_cls, objectness_probs):
                queries = center_unnormalized_each[(pred_sem_cls_each < dataset_config.num_semcls) & (objectness_probs_each > 0.05)] #TODO magic number
                queries = queries.cpu()
                batches.append(queries)
            last_prev_detections = [detections.numpy() for detections in prev_detections]
            prev_detections = batches

            batch_pred_map_cls = parse_predictions(
                predicted_box_corners,
                sem_cls_probs,
                objectness_probs,
                point_cloud,
                ap_config_dict,
            )

            point_cloud = point_cloud.cpu().detach().numpy()
            query_xyz = query_xyz.cpu().detach().numpy()

            point_cloud = flip_axis_to_camera_np(point_cloud)
            last_prev_detections = [flip_axis_to_camera_np(detections) for detections in last_prev_detections]
            query_xyz = np.stack([flip_axis_to_camera_np(queries) for queries in query_xyz])

            renderer.draw_point_cloud(point_cloud[0])
            for query in query_xyz[0]:
                renderer.draw_sphere(query, color=[0.2, 0.4, 0.2])
            for detection in last_prev_detections[0]:
                renderer.draw_sphere(detection, size=0.3, color=[1.0, 0.0, 0.0])
            for box in batch_pred_map_cls[0]:
                renderer.draw_box(box[1])
            filepath = path.join(args.render_output, f'{batch_idx}.png')
            print(filepath)
            renderer.render_image(filepath)
            renderer.clear_scene()
    elif (args.render_kitti_dataset == 'kitti-frame'):
        for batch_idx, batch_data_label in enumerate(dataset_loader):
            curr_time = time.time()

            batch_cnt = len(batch_data_label['point_clouds'])

            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(net_device)

            inputs = {
                "point_clouds": batch_data_label["point_clouds"],
                "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
                "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
            }
            outputs, query_xyz = model(inputs, return_queries=True)

            # Memory intensive as it gathers point cloud GT tensor across all ranks
            outputs["outputs"] = all_gather_dict(outputs["outputs"])
            inputs = all_gather_dict(inputs)

            predicted_box_corners=outputs['outputs']["box_corners"]
            sem_cls_probs=outputs['outputs']["sem_cls_prob"]
            objectness_probs=outputs['outputs']["objectness_prob"]
            point_cloud=batch_data_label["point_clouds"]

            batch_pred_map_cls = parse_predictions(
                predicted_box_corners,
                sem_cls_probs,
                objectness_probs,
                point_cloud,
                ap_config_dict,
            )

            point_cloud = point_cloud.cpu().detach().numpy()
            query_xyz = query_xyz.cpu().detach().numpy()

            point_cloud = flip_axis_to_camera_np(point_cloud)
            query_xyz = np.stack([flip_axis_to_camera_np(queries) for queries in query_xyz])

            gt_box_corners=batch_data_label["gt_box_corners"]
            gt_box_sem_cls_labels=batch_data_label["gt_box_sem_cls_label"]
            gt_box_present=batch_data_label["gt_box_present"]

            #NOTE This code is copied from ap_calculator.step()
            gt_box_corners = gt_box_corners.cpu().detach().numpy()
            gt_box_sem_cls_labels = gt_box_sem_cls_labels.cpu().detach().numpy()
            gt_box_real = gt_box_sem_cls_labels != dataset_config.num_semcls
            gt_box_present = gt_box_present.cpu().detach().numpy()

            renderer.draw_point_cloud(point_cloud[0])
            for gt_box in gt_box_corners[0][gt_box_real[0]]:
                renderer.draw_box(gt_box, color=[0.0, 0.5, 0.0])
            for query in query_xyz[0]:
                renderer.draw_sphere(query, color=[0.2, 0.4, 0.2])
            for box in batch_pred_map_cls[0]:
                renderer.draw_box(box[1])
            filepath = path.join(args.render_output, f'{batch_idx}.png')
            print(filepath)
            renderer.render_image(filepath)
            renderer.clear_scene()

        barrier()