from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from importlib_metadata import Sectioned

import _init_paths
import os
import os.path as osp
import cv2
import logging
import numpy as np

# from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.utils import *
from collections import defaultdict
from deep_sort.mvtracker import MVTracker
from deep_sort.update import Update
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sv_threshold', type=float, default=0.3, help="single view threshold")
parser.add_argument('--cv_threshold', type=float, default=0.3, help="cross view threshold")
parser.add_argument('--feature', type=str, default=None, help="feature npy file")
parser.add_argument('--result_dir', type=str, default="result", help="result directory")
args = parser.parse_args()

def search_feature(frame_id, view_id, feature_list):
    for item in feature_list:
        if frame_id == item[0] and view_id == item[1]:
            return item[2:]
    return []

def gather_seq_info_multi_view(view_ls, datas, seq):
    seq_dict = {}
    print('loading dataset...')

    det_data = datas[seq]

    detections = defaultdict(list)
    view_detections = defaultdict(list)

    for view in view_ls:
        det_data[view] = sorted(det_data[view], key=lambda x:x[0])

    min_frame, max_frame = det_data[view_ls[0]][0][0], det_data[view_ls[0]][-1][0]
    for view in view_ls:
        '''
        data: frame_id, view_id, xmin, ymin, w, h, score, 0, 0, 0, single_view_feature, cross_view_feature
        frame_id : 1, 2, 3, ...
        view_id: 0, 1 , 2, ...
        single_view_feature: shape (512)
        cross_view_feature: shape (512)
        '''
        det_data[view] = sorted(det_data[view], key=lambda x:x[0])
        for data in det_data[view]:
            data[1] = -1
            detections[view].append(data)
            view_detections[view].append(data)
            
    for view in view_ls:
        seq_dict[view] = { 
            "image_filenames": seq,
            "detections": np.array(detections[view]),
            "view_detections": np.array(view_detections[view]),
            "image_size":(3, 1920, 1080),
            "min_frame_idx": int(min_frame),
            "max_frame_idx": int(max_frame) }
    return seq_dict

def main(datas, result_root, seqs):
    logger.setLevel(logging.INFO)
    mkdir_if_missing(result_root)
    # run tracking
    for seq in seqs:
        logger.info('start seq: {}'.format(seq))

        view_ls = ['Drone', 'View1', 'View2']
        seq_mv = gather_seq_info_multi_view(view_ls, datas, seq)

        mvtracker = MVTracker(view_ls, args.sv_threshold)
        updater = Update(seq=seq_mv, 
                         mvtracker=mvtracker, 
                         display=0, 
                         view_list=view_ls, 
                         cv_threshold=args.cv_threshold,
                         )
        updater.run()

        for view in view_ls:
            if not os.path.exists(os.path.join(result_root, seq)):
                os.mkdir(os.path.join(result_root, seq))
            f = open(os.path.join(result_root, seq, '{}.txt'.format(view)), 'w')
            for row in updater.result[view]:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                    row[0], row[1], row[2], row[3], row[4], row[5]), file=f)
            f.close() 


if __name__ == '__main__':
    seqs_str = '''circleRegion
                  innerShop
                  movingView
                  park
                  playground
                  shopFrontGate
                  shopSecondFloor
                  shopSideGate
                  shopSideSquare
                  southGate'''
    seqs = [seq.strip() for seq in seqs_str.split()]

    datas = np.load(args.feature, allow_pickle=True).item()

    main(datas, args.result_dir, seqs)
