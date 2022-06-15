from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import numpy as np

from tracking_utils.log import logger
from tracking_utils.timer import Timer
import datasets.dataset.jde as datasets
import torch
from tracking_utils.utils import mkdir_if_missing, xyxy2xywh
from opts import opts
from models.decode import mot_decode
from utils.post_process import ctdet_post_process
from models.model import create_model, load_model


def write_results_score(filename, results):
    save_format = '{frame},{person_id},{x1},{y1},{w},{h},{s},{ep1},{ep2},{ep3}\n'
    with open(filename, 'w') as f:
        for frame_id, person_id, tlbrs, scores, ep1, ep2, ep3 in results:
            for tlbr in tlbrs:
                x1, y1, w, h = tlbr
                line = save_format.format(frame=frame_id, person_id=person_id, x1=x1, y1=y1, w=w, h=h, s=scores, ep1=ep1, ep2=ep2, ep3=ep3)
                f.write(line)
    print('save results to {}'.format(filename))


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
    if len(scores) > 128:
        kth = len(scores) - 128
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def eval_seq(opt, dataloader, datatype, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    # model = torch.nn.DataParallel(model)
    model = model.to(opt.device)
    model.eval()
    timer = Timer()
    results = []
    frame_id = 0
    for path, img, img0 in dataloader:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # run detecting
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = blob.shape[2]
        inp_width = blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // opt.down_ratio,
                'out_width': inp_width // opt.down_ratio}
        with torch.no_grad():
            output = model(blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=opt.ltrb, K=opt.K)

        dets = post_process(opt, dets, meta)
        dets = merge_outputs(opt, [dets])[1]

        dets = dets[dets[:, 4] > opt.conf_thres]
        # dets[:, :4] = xyxy2xywh(dets[:, :4])

        tlbrs = []
        scores = []
        for *tlbr, conf in dets:
            tlbrs.append(tlbr)
            scores.append(conf)
        timer.toc()
        # save results
        results.append((frame_id + 1, -1, tlbrs, 1, -1, -1, -1))
        frame_id += 1
    # save results
    write_results_score(result_filename, results)
    #write_results_score_hie(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'dets', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))

        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    seqs_str = '''circleRegion_Drone
                  circleRegion_View1
                  circleRegion_View2
                  innerShop_Drone
                  innerShop_View1
                  innerShop_View2
                  movingView_Drone
                  movingView_View1
                  movingView_View2
                  park_Drone
                  park_View1
                  park_View2
                  playground_Drone
                  playground_View1
                  playground_View2
                  shopFrontGate_Drone
                  shopFrontGate_View1
                  shopFrontGate_View2
                  shopSecondFloor_Drone
                  shopSecondFloor_View1
                  shopSecondFloor_View2
                  shopSideGate_Drone
                  shopSideGate_View1
                  shopSideGate_View2
                  shopSideSquare_Drone
                  shopSideSquare_View1
                  shopSideSquare_View2
                  southGate_Drone
                  southGate_View1
                  southGate_View2 '''
    data_root = '../../datasets/DIVO/images/test/'
    seqs = [seq.strip() for seq in seqs_str.split()]
    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='detection_results',
         show_image=False,
         save_images=False,
         save_videos=False)
