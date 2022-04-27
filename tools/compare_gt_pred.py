''' Utility to make the evaluation script for the KITTI 3D Object benchmark usable with training split output '''

import os
import argparse
import shutil

parser = argparse.ArgumentParser('Compare', add_help=False)

parser.add_argument('--gt_folder', default="./kitti/label_2/", 
                    type=str, help="Ground Truth Folder")
parser.add_argument('--pred_folder', default="./kitti/pred/predictions_fnet4/", 
                    type=str, help="Predictions Folder")
parser.add_argument('--pred_output', type=str, help="Where to output missing predictions")
parser.add_argument('--gt_output', type=str, help="Where to output blank unmatched gt")

args = parser.parse_args()

folder_one = args.gt_folder
folder_two = args.pred_folder
pred_output = args.pred_output
gt_output = args.gt_output

if pred_output != None and not os.path.isdir(pred_output):
    os.mkdir(pred_output)
if gt_output != None and not os.path.isdir(gt_output):
    os.mkdir(gt_output)

for i in range(7518):
    filename = f'{i:06d}.txt'
    print(filename)
    if os.path.exists(folder_two + filename):
        if pred_output != None:
            shutil.copyfile(folder_two + filename, pred_output + filename)
            print("Created: " + pred_output + filename)
        if gt_output != None:
            shutil.copyfile(folder_one + filename, gt_output + filename)
            print("Created: " + gt_output + filename)
    else:
        if pred_output != None:
            with open(pred_output + filename, 'w') as f:
                print("Created: " + pred_output + filename)
        if gt_output != None:
            with open(gt_output + filename, 'w') as f:
                print("Created: " + gt_output + filename)