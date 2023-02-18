from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts


from utils.post_process import ctdet_post_process
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.image import get_affine_transform
from collections import defaultdict
from tqdm import tqdm
from deep_sort.mvtracker import MVTracker
from deep_sort.update import Update
import pdb
    
def post_process(opt, dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], opt.num_classes)
    for j in range(1, opt.num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]

def merge_outputs(opt, detections):
    results = {}
    for j in range(1, opt.num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, opt.num_classes + 1)])
    if len(scores) > opt.K:
        kth = len(scores) - opt.K
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results

def gather_seq_info_multi_view(opt, dataloader, seq, seq_length,use_cuda = True):
    seq_dict = {}
    # print('loading dataset...')

    image_filenames = defaultdict(list)
    detections = defaultdict(list)
    view_detections = defaultdict(list)
    #model     
    # print('Creating model...')
    if opt.gpus[0] >= 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(device)
    model.eval()

    view_ls = dataloader.view_list
    for data_i, (path, img, img0) in tqdm(enumerate(dataloader), total=len(dataloader)):

        #blob
        view = path.split('/')[-1].split('_')[1]

        frame_index = int(path.split('/')[-1].split('.jpg')[0].split('_')[-1])
        #int(scn[0].split('_')[-1].split('.jpg')[0])
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = blob.shape[2]
        inp_width = blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // opt.down_ratio,
                'out_width': inp_width // opt.down_ratio}

        #output
        with torch.no_grad():
            output = model(blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']

            if opt.baseline == 0:
                view_id_feature = F.normalize(output['cross_view_id'], dim = 1)
                id_feature = F.normalize(output['single_view_id'], dim = 1)
            else:
                if opt.baseline_view == 0:
                    id_feature = F.normalize(output['single_view_id'], dim = 1)
                else:
                    view_id_feature = F.normalize(output['cross_view_id'], dim = 1)

            reg = output['reg'] if opt.reg_offset else None
            dets, bboxes, scores, clses, inds = mot_decode(hm, wh, reg=reg, ltrb=opt.ltrb, K=opt.K)

            if opt.baseline == 0:
                id_feature = _tranpose_and_gather_feat(id_feature, inds).squeeze(0).cpu().numpy()
                view_id_feature = _tranpose_and_gather_feat(view_id_feature, inds).squeeze(0).cpu().numpy()
            else:
                if opt.baseline_view == 0:
                    id_feature = _tranpose_and_gather_feat(id_feature, inds).squeeze(0).cpu().numpy()
                    view_id_feature = id_feature
                else:
                    view_id_feature = _tranpose_and_gather_feat(view_id_feature, inds).squeeze(0).cpu().numpy()
                    id_feature = view_id_feature
        #detections   

        dets = post_process(opt, dets, meta)
        dets = merge_outputs(opt, [dets])[1]
        remain_inds = dets[:, 4] > opt.conf_thres
        dets = dets[remain_inds]
        bboxes = bboxes[0][remain_inds]
        scores = scores[0][remain_inds]
        id_feature = id_feature[remain_inds]
        view_id_feature = view_id_feature[remain_inds]

        for feature, view_feature, detection, id in zip(id_feature, view_id_feature, dets, remain_inds):
            index = frame_index
            confidence = detection[-1]
            detection = [int(i) for i in detection]
            #id = int(id[0])
            confidence = confidence.item()
            det = [index] + [id] + [detection[0], detection[1], detection[2] - detection[0], detection[3] - detection[1]] + [confidence] + [0, 0, 0] + feature.tolist()
            detections[view].append(det)
            view_det = [index] + [id] + [detection[0], detection[1], detection[2] - detection[0], detection[3] - detection[1]] + [confidence] + [0, 0, 0] + view_feature.tolist()
            view_detections[view].append(view_det)
               
    for view in view_ls:
        view_dict = { 
                "image_filenames": seq,
                "detections": np.array(detections[view]),
                "view_detections": np.array(view_detections[view]),
                "image_size":(3, 1920, 1080),
                "min_frame_idx": 1,
                "max_frame_idx": seq_length}
        if opt.test_divo:
            seq_dict = view_dict
        if opt.test_mvmhat or opt.test_mvmhat_campus or opt.test_wildtrack:
            view_dict["min_frame_idx"] = int(seq_length * 2 / 3) + 1
            seq_dict[view] = view_dict
        if opt.test_epfl:
            view_dict["min_frame_idx"] = int(seq_length)
            view_dict["max_frame_idx"] = int(seq_length * 3 / 2)
    return seq_dict

def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=True, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    view_ls = ['View1', 'View2', 'View3']

    # run tracking
    for seq in seqs:
        logger.info('start seq: {}'.format(seq))
        if opt.test_divo:
            seq_mv = {}
            for view in view_ls:
                dataloader = datasets.LoadImages_DIVO(opt, osp.join(data_root, '{}_{}'.format(seq, view), 'img1'), opt.img_size)
                seq_mv[view] = gather_seq_info_multi_view(opt, dataloader, seq, dataloader.seq_length)
        else:
            dataloader = datasets.LoadImages(opt, osp.join(data_root, seq), opt.img_size)
            meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
            seq_length = int(meta_info[meta_info.find('seqLength=') + 10:meta_info.find('\nimWidth')])
            seq_mv = gather_seq_info_multi_view(opt, dataloader, seq, seq_length)
            view_ls = dataloader.view_list
        
        mvtracker = MVTracker(view_ls)
        updater = Update(opt, seq=seq_mv, mvtracker=mvtracker, display=0, view_list=view_ls)
        updater.run()

        for view in view_ls:
            if not os.path.exists(os.path.join(result_root, seq)):
                os.mkdir(os.path.join(result_root, seq))
            f = open(os.path.join(result_root, seq, '{}.txt'.format(view)), 'w')
            #f = open(result_filename, 'w')
            for row in updater.result[view]:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                    row[0], row[1], row[2], row[3], row[4], row[5]), file=f)
            f.close() 


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
        
    if opt.test_divo:
        seqs_str = '''Circle
                      Shop
                      Moving
                      Park
                      Ground
                      Gate1
                      Floor
                      Side
                      Square
                      Gate2'''
        data_root = os.path.join(opt.data_dir, 'DIVOTrack/images/test')

    if opt.test_mvmhat:
        seqs_str = '''scene1
                      scene2
                      scene3
                      scene4
                      scene5
                      scene6'''
        data_root = os.path.join(opt.data_dir, 'FairMOT_MVMHAT/images/train') 

    if opt.test_mvmhat_campus:
        seqs_str = '''scene1
                      scene2
                      scene3'''
        data_root = os.path.join(opt.data_dir, 'FairMOT_MVMHAT_campus/images/train') 

    if opt.test_wildtrack:
        seqs_str = '''wildtrack'''
        data_root = os.path.join(opt.data_dir, 'wildtrack/images/train') 
 
    if opt.test_epfl:
        seqs_str = '''basketball
                      laboratary
                      passageway
                      terrace'''
        data_root = os.path.join(opt.data_dir, 'CrossMOT_dataset/EPFL/images/train')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root = data_root,
         seqs = seqs,
         exp_name = opt.exp_name,
         show_image = True,
         save_images = True,
         save_videos = False)
