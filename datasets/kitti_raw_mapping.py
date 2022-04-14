import os
import tarfile
import numpy as np

# TODO add setup instructions for compressed raw dataset?

class KittiRawMapping:
    def __init__(self, root_dir):
        indices_for_mappings = None
        with open(os.path.join(root_dir, 'mapping', 'train_rand.txt'), 'r') as inp:
            indices_for_mappings = [int(value)-1 for value in inp.read().split(',')]
        raw_mapping_lines = None
        with open(os.path.join(root_dir, 'mapping', 'train_mapping.txt'), 'r') as inp:
            raw_mapping_lines = [line.split(' ') for line in inp]
        mappings_to_raw = [(os.path.join(root_dir, 'raw', line[0], f'{line[1]}.tar.gz'), './velodyne_points/data', int(line[2])) for line in raw_mapping_lines]
        self.raw_mapping = [mappings_to_raw[index] for index in indices_for_mappings]

    def load_velo_scans_from_compressed(file_path, local_paths):
        velo_scans = []
        with tarfile.open(file_path, "r:gz") as targz:
            for local_path in local_paths:
                with targz.extractfile(local_path) as velo_file:
                    #see kitti_util.load_velo_scan
                    scan = np.frombuffer(velo_file.read(), dtype=np.float32)
                    scan = scan.reshape((-1, 4))
                    velo_scans.append(scan)
        return velo_scans