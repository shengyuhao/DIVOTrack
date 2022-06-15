# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
import pdb
import numpy as np
from tqdm import tqdm

def search_bbox(fid, pid, det_list):
    for item in det_list:
        if int(fid) == int(item[0]) and int(pid) == int(item[1]):
            return [int(item[2]), int(item[3]), int(item[4]-item[2]), int(item[5] - item[3])]
    return []

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    

    save_dict = {}
    det_dict = {}

    seq_list = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
    view_list = ['Drone', 'View1', 'View2']

    res_dir = '../../datasets/DIVO/images/dets/detection_results/'

    for seq in seq_list:
        save_dict[seq] = {}
        det_dict[seq] = {}
        for view in view_list:
            save_dict[seq][view] = []
            det_dict[seq][view] = np.loadtxt(os.path.join(res_dir, '{}_{}.txt'.format(seq, view)), delimiter=',').tolist()

    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)
    model.cuda()

    for data in tqdm(val_loader):
        model.eval()
        feature = model(data[0].unsqueeze(0).cuda()).detach().cpu().numpy().tolist()
        pid= int(data[3].split('/')[-1].split('_')[0])
        fid = int(data[3].split('/')[-1].split('_')[-1].split('.')[0][1:])
        seq = seq_list[int(data[3].split('/')[-1].split('_')[1][3:]) - 1]
        view = view_list[int(data[3].split('/')[-1].split('_')[1][1]) - 1]
        bbox = search_bbox(fid, pid, det_dict[seq][view])
        save_dict[seq][view].append([fid] + [pid] + bbox + [1, 0, 0, 0] + feature[0])


    np.save('./rsb_self.npy', save_dict)

if __name__ == '__main__':
    main()

# python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.RE_RANKING "('yes')" TEST.WEIGHT "('../models/resnet50_model_120.pth')"