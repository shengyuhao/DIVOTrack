import os.path as osp
import os
import numpy as np
from tqdm import tqdm

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = '/mnt/sdb/dataset/MOT_datasets/DIVOTrack/images/train'
label_root = '/mnt/sdb/dataset/MOT_datasets/DIVOTrack/labels_with_ids/train'
gt_root = '/mnt/sdb/dataset/MOT_datasets/annotation/test_gt'
mkdirs(label_root)
#seqs = [s for s in os.listdir(seq_root)]
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

tid_curr = 0
tid_last = -1
for seq in tqdm(seqs):
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    seq_label_root = osp.join(label_root, seq)
    mkdirs(seq_label_root)
    for view in views:
        gt_txt = osp.join(gt_root, seq, view + '.txt')
        gt = np.genfromtxt(gt_txt, dtype=np.float64, delimiter=' ')
    # idx = np.lexsort(gt.T[:2, :])
    # gt = gt[idx, :]
        for fid, tid, lx, ly, rx, ry in gt:
            if view == 'View1':
                lx *= (1920 / 3640)
                rx *= (1920 / 3640)
                ly *= (1080 / 2048)
                ry *= (1080 / 2048)
            fid = int(fid)
            tid = int(tid)
            x = (lx + rx) / 2
            y = (ly + ry) / 2
            w = rx - lx
            h = ry - ly
            label_fpath = osp.join(seq_label_root, '{}_{}_{:06d}.txt'.format(seq, view, fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
        
