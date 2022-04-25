# Copyright (c) Facebook, Inc. and its affiliates.
from .kitti import KITTI3DObjectDetectionDataset, KITTI3DObjectDetectionDatasetConfig
from .scannet import ScannetDetectionDataset, ScannetDatasetConfig
from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig


DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
    "kitti": [KITTI3DObjectDetectionDataset, KITTI3DObjectDetectionDatasetConfig]
}

def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
    
    if args.render_only:
        dataset_dict = {
            "train": dataset_builder(
                dataset_config, 
                split_set=\
                    "train-clip" if args.render_kitti_dataset == 'kitti-clip' else \
                        ("train" if args.render_kitti_dataset == 'kitti-frame' else "video"), 
                root_dir=args.dataset_root_dir, 
                augment=False, 
            ),
            "test": None,
        }
    elif args.predict_only:
        dataset_dict = {
            "train": dataset_builder(
                dataset_config, 
                split_set="train-clip", 
                root_dir=args.dataset_root_dir, 
                augment=False, 
            ),
            "test": dataset_builder(
                dataset_config, 
                split_set="val-clip", 
                root_dir=args.dataset_root_dir, 
                augment=False, 
            ),
        }
    else:
        dataset_dict = {
            "train": dataset_builder(dataset_config, split_set="train", root_dir=args.dataset_root_dir, augment=True),
            "test": dataset_builder(dataset_config, split_set="val", root_dir=args.dataset_root_dir, augment=False),
        }
    return dataset_dict, dataset_config
    