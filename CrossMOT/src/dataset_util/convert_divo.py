import os.path as osp
import os
import numpy as np
import pdb
from tqdm import tqdm

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


ori_label_root = '/mnt/sdb/dataset/MOT_datasets/CrossMOT_dataset/DIVOTrack/labels_with_ids/test'
tar_label_root = '/mnt/sdb/dataset/MOT_datasets/DIVOTrack/labels_with_ids_cross_view/test'
mkdirs(tar_label_root)
seqs = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
views = ['Drone', 'View1', 'View2']

seqs_dict = {'circleRegion': 'Circle',
             'innerShop': 'Shop',
             'movingView': 'Moving',
             'park': 'Park',
             'playground': 'Ground',
             'shopFrontGate': 'Gate1',
             'shopSecondFloor': 'Floor',
             'shopSideGate': 'Side',
             'shopSideSquare': 'Square',
             'southGate': 'Gate2'}

views_dict = {'Drone': 'View1', 'View1': 'View2', 'View2': 'View3'}

min_frame = 0
base_person_id = 0
max_person_id = 0

for seq in seqs:
    ori_txt_path = osp.join(ori_label_root, seq)
    ori_txt_list = sorted(os.listdir(ori_txt_path))
    for i in ori_txt_list:
        if not '.txt' in i:
            ori_txt_list.remove(i)
    min_frame = int(ori_txt_list[0].split('_')[-1].split('.')[0])
    # base_person_id += max_person_id
    # max_person_id = 0
    for view in views:
        seq_label_root = osp.join(tar_label_root, '{}_{}'.format(seqs_dict[seq], views_dict[view]))
        mkdirs(seq_label_root)
        seq_label_root = osp.join(seq_label_root, 'img1')
        mkdirs(seq_label_root)
        for ori_txt_file in tqdm(ori_txt_list):
            if view in ori_txt_file and '.txt' in ori_txt_file:
                txt = osp.join(ori_txt_path, ori_txt_file)
                gt = np.genfromtxt(txt, dtype=np.float64, delimiter=' ')
                if gt.size <= 6:
                    gt = gt.reshape(1, 6)
                cur_frame = int(ori_txt_file.split('_')[-1].split('.')[0])
                try:
                    for fid, tid, a, b, c, d in gt:
                        label_fpath = osp.join(seq_label_root, '{}_{}_{:06d}.txt'.format(seqs_dict[seq], views_dict[view], cur_frame - min_frame + 1))
                        # pid = int(tid) + base_person_id
                        # if max_person_id < pid:
                        #     max_person_id = pid
                        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(int(tid), a, b, c, d)
                        with open(label_fpath, 'a') as f:
                            f.write(label_str)
                except:
                    pdb.set_trace()
        # gt = np.genfromtxt(gt_txt, dtype=np.float64, delimiter=' ')
        # for fid, tid, lx, ly, rx, ry in gt:
        #     fid = int(fid)
        #     tid = int(tid)
        #     x = (lx + rx) / 2
        #     y = (ly + ry) / 2
        #     w = rx - lx
        #     h = ry - ly
        #     label_fpath = osp.join(seq_label_root, '{}_{}_{:06d}.txt'.format(seq, view, fid))
        #     label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
        #         tid, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        #     with open(label_fpath, 'a') as f:
        #         f.write(label_str)
        
