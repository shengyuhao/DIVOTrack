import os
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm


def mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)
splits = ['train', 'test']
output_dir = '/mnt/sdb/mvmhat' 
root_dir = "../../datasets/DIVO/images"
scene_ls = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
mkdir(output_dir)
print("The dataset will be save to {}".format(output_dir))
for split in splits:
    print("Converting {}".format(split))
    mkdir(osp.join(output_dir, 'images'))
    mkdir(osp.join(output_dir, '{}_gt'.format(split)))
    split_dir = osp.join(root_dir, split)
    for scene in tqdm(sorted(os.listdir(split_dir))):
        mkdir(osp.join(output_dir, 'images', scene.split('_')[0]))
        mkdir(osp.join(output_dir, '{}_gt'.format(split), scene.split('_')[0]))
        img_list = sorted(os.listdir(osp.join(split_dir, scene, 'img1')))
        for img in tqdm(img_list):
            img_path = osp.join(split_dir, scene, 'img1', img)
            cv2.imwrite(osp.join(output_dir, 'images', scene.split('_')[0], img), cv2.imread(img_path))
        gt_file = sorted(np.loadtxt(osp.join(split_dir, scene, 'gt/gt.txt'), delimiter=',').tolist(), key=lambda x:x[0])
        gt_file = np.array(gt_file)
        gt_file[:,4] += gt_file[:,2]
        gt_file[:,5] += gt_file[:,3]
        gt_file = np.delete(gt_file, -1, 1)
        gt_file = np.delete(gt_file, -1, 1)
        gt_file = np.delete(gt_file, -1, 1)
        np.savetxt(osp.join(output_dir, '{}_gt'.format(split), scene.split('_')[0], "{}.txt".format(scene.split('_')[1])), gt_file, fmt="%d", delimiter=' ')