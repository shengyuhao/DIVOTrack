# encoding: utf-8
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine

from tqdm import tqdm
import numpy as np
import os

def create_supervised_evaluator(model, metrics, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to evaluate
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def do_test(
        cfg,
        model,
        data_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline")
    logger.info("Enter inferencing")
    model.eval()
    save_dict = {}

    seq_list = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
    view_list = ['Drone', 'View1', 'View2']

    import json
    filename = '../../datasets/DIVO/boxes.json'
    boxes = json.load(open(filename))
    for seq in seq_list:
        save_dict[seq] = {}
        for view in view_list:
            save_dict[seq][view] = []
    for data in tqdm(data_loader):
        feature = model(data[0].unsqueeze(0).cuda()).tolist()
        pid= int(data[3].split('/')[-1].split('_')[0])
        fid = int(data[3].split('/')[-1].split('_')[-1].split('.')[0][1:])
        seq = seq_list[int(data[3].split('/')[-1].split('_')[1][3:]) - 1]
        view = view_list[int(data[3].split('/')[-1].split('_')[1][1]) - 1]
        img = data[3].split('/')[-1]
        bbox_str = boxes[img]
        xmin = int(bbox_str.split(',')[0])
        ymin = int(bbox_str.split(',')[1])
        xmax = int(bbox_str.split(',')[2])
        ymax = int(bbox_str.split(',')[3])
        save_dict[seq][view].append([fid] + [pid] + [xmin,ymin,xmax-xmin,ymax-ymin] + [1, 0, 0, 0] + feature[0])        
        #save_dict[seq][view].append([fid] + [pid] + bbox + [1, 0, 0, 0] + feature[0])
    np.save('../../datasets/DIVO/npy/cross_view/AGW/AGW.npy', save_dict)
