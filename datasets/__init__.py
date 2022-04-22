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
    
    if args.sample_raw_only:
        dataset_dict = {
            "train": dataset_builder(
                dataset_config, 
                split_set="train-clip", 
                root_dir=args.dataset_root_dir, 
                augment=False, 
            ),
            "test": None,
        }
    else:
        dataset_dict = {
            "train": dataset_builder(dataset_config, split_set="train", root_dir=args.dataset_root_dir, augment=True),
            "test": dataset_builder(dataset_config, split_set="val", root_dir=args.dataset_root_dir, augment=False),
        }
    return dataset_dict, dataset_config
    