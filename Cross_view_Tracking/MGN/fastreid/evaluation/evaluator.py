# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from contextlib import contextmanager

import torch

from fastreid.utils import comm
from fastreid.utils.logger import log_every_n_seconds
import numpy as np

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.
    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def preprocess_inputs(self, inputs):
        pass

    def process(self, inputs, outputs):
        """
        Process an input/output pair.
        Args:
            inputs: the inputs that's used to call the model.
            outputs: the return value of `model(input)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:
                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


# class DatasetEvaluators(DatasetEvaluator):
#     def __init__(self, evaluators):
#         assert len(evaluators)
#         super().__init__()
#         self._evaluators = evaluators
#
#     def reset(self):
#         for evaluator in self._evaluators:
#             evaluator.reset()
#
#     def process(self, input, output):
#         for evaluator in self._evaluators:
#             evaluator.process(input, output)
#
#     def evaluate(self):
#         results = OrderedDict()
#         for evaluator in self._evaluators:
#             result = evaluator.evaluate()
#             if is_main_process() and result is not None:
#                 for k, v in result.items():
#                     assert (
#                             k not in results
#                     ), "Different evaluators produce results with the same key {}".format(k)
#                     results[k] = v
#         return results


def inference_on_dataset(model, data_loader, evaluator, flip_test=False):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
        flip_test (bool): If get features with flipped images
    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader.dataset)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    

    
    save_dict = {}
    from glob import glob
    import json
    
    '''
    # wildtrack version
    img_root = '/mnt/sdb/dataset/MOT_datasets/CrossMOT_dataset/wildtrack/ReID_format/bounding_box_test/'
    file_list = glob('/mnt/sdb/dataset/MOT_datasets/CrossMOT_dataset/wildtrack/images/dets/det_res/*.txt')
    file_list.sort()
    seq_list = ['wildtrack']

    view_list = ['view1', 'view2', 'view3', 'view4', 'view5', 'view6', 'view7']
    boxes={}
    for file in file_list:
        dets = np.loadtxt(file, delimiter=',').tolist()
        scene = file.split('/')[-1].split('_')[0]
        cam = file.split('/')[-1].split('_')[-1].split('.txt')[0]
        for det in dets:
            if int(float(det[0])) >= 268:
                img_path = '{}_c{}s{}_f{}.jpg'.format(str(int(det[1])).zfill(4), cam[-1], seq_list.index(scene)+1, str(int(det[0])-1).zfill(6))
                boxes[img_path] = '{},{},{},{}'.format(int(det[2]), int(det[3]), int(det[4]), int(det[5]))


    for seq in seq_list:
        save_dict[seq] = {}
        for view in view_list:
            save_dict[seq][view] = []      

    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            # Flip test
            if flip_test:
                inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = model(inputs)
                outputs = (outputs + flip_outputs) / 2
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )

            for ind in range(0, len(inputs['images'])):
                feature = outputs[ind].cuda().tolist()
                pid = -1
                img_path = inputs['img_paths'][ind]
                fid = int(img_path.split('/')[-1].split('f')[-1][:6])
                if fid <= 266:
                    continue
                seq = seq_list[int(img_path.split('/')[-1].split('s')[-1].split('_')[0])-1]
                view = view_list[int(img_path.split('/')[-1].split('c')[-1][0]) - 1]
                #import pdb;pdb.set_trace()
                bbox_str = boxes[img_path.split('/')[-1]].split(',')
                
                xmin = int(bbox_str[0])
                ymin = int(bbox_str[1])
                xmax = int(bbox_str[2])
                ymax = int(bbox_str[3])
                save_dict[seq][view].append([fid] + [pid] + [xmin,ymin,xmax-xmin,ymax-ymin] + [1, 0, 0, 0] + feature) # 2048   
            
        np.save('./MGN_wildtrack.npy', save_dict)
    return None    
    '''
    
    
    '''
    # epfl version
    img_root = '/mnt/sdb/dataset/MOT_datasets/CrossMOT_dataset/EPFL/ReID_format/bounding_box_test/'
    file_list = glob('/mnt/sdb/dataset/MOT_datasets/CrossMOT_dataset/EPFL/images/dets/det_res/*.txt')
    file_list.sort()
    seq_list = ['basketball', 'laboratary', 'passageway', 'terrace']

    view_list = ['view1', 'view2', 'view3', 'view4']
    boxes={}
    for file in file_list:
        dets = np.loadtxt(file, delimiter=',').tolist()
        scene = file.split('/')[-1].split('_')[0]
        cam = file.split('/')[-1].split('_')[-1].split('.txt')[0]
        for det in dets:
            img_path = '{}_c{}s{}_f{}.jpg'.format(str(int(det[1])).zfill(4), cam[-1], seq_list.index(scene)+1, str(int(det[0])).zfill(6))
            boxes[img_path] = '{},{},{},{}'.format(int(det[2]), int(det[3]), int(det[4]), int(det[5]))
    

    for seq in seq_list:
        save_dict[seq] = {}
        if seq == 'Basketball':
            for view in view_list[:3]:
                save_dict[seq][view] = []
        else:
            for view in view_list:
                save_dict[seq][view] = []     

    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            # Flip test
            if flip_test:
                inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = model(inputs)
                outputs = (outputs + flip_outputs) / 2
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )

            for ind in range(0, len(inputs['images'])):
                feature = outputs[ind].cuda().tolist()
                pid = -1
                img_path = inputs['img_paths'][ind]
                fid = int(img_path.split('/')[-1].split('f')[-1][:6])
                seq = seq_list[int(img_path.split('/')[-1].split('s')[-1].split('_')[0])-1]
                view = view_list[int(img_path.split('/')[-1].split('c')[-1][0]) - 1]
                
                bbox_str = boxes[img_path.split('/')[-1]].split(',')
                
                xmin = int(bbox_str[0])
                ymin = int(bbox_str[1])
                xmax = int(bbox_str[2])
                ymax = int(bbox_str[3])
                save_dict[seq][view].append([fid] + [pid] + [xmin,ymin,xmax-xmin,ymax-ymin] + [1, 0, 0, 0] + feature) # 2048   
            
        np.save('./MGN_epfl.npy', save_dict)
    return None    
    '''
    
    '''
    # mm version
    img_root = '/mnt/sdb/dataset/MOT_datasets/CrossMOT_dataset/MvMHAT/ReID_format/bounding_box_test/'
    file_list = glob('/home/xqr/shengyuhao/ReID-Survey/toDataset/mm/detection_res_test/*.txt')
    file_list.sort()
    seq_list = ['scene1', 'scene2','scene3', 'scene4','scene5', 'scene6']

    view_list = ['view1', 'view2', 'view3', 'view4']
    boxes={}
    for file in file_list:
        dets = np.loadtxt(file, delimiter=',').tolist()
        scene = file.split('/')[-1].split('_')[0]
        cam = file.split('/')[-1].split('_')[-1].split('.txt')[0]
        for det in dets:
            img_path = '{}_c{}s{}_f{}.jpg'.format(str(int(det[1])).zfill(4), cam[-1], scene[-1], str(int(det[0])).zfill(6))
            boxes[img_path] = '{},{},{},{}'.format(int(det[2]), int(det[3]), int(det[4]), int(det[5]))
    #import pdb;pdb.set_trace()

    for seq in seq_list:
        save_dict[seq] = {}
        if seq == 'scene6':
            for view in view_list[:3]:
                save_dict[seq][view] = []
        else:
            for view in view_list:
                save_dict[seq][view] = []
    
    
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            # Flip test
            if flip_test:
                inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = model(inputs)
                outputs = (outputs + flip_outputs) / 2
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )

            for ind in range(0, len(inputs['images'])):
                feature = outputs[ind].cuda().tolist()
                pid = -1
                img_path = inputs['img_paths'][ind]
                fid = int(img_path.split('/')[-1].split('f')[-1][:6])
                seq = seq_list[int(img_path.split('/')[-1].split('s')[-1].split('_')[0])-1]
                view = view_list[int(img_path.split('/')[-1].split('c')[-1][0]) - 1]
                
                bbox_str = boxes[img_path.split('/')[-1]].split(',')
                
                xmin = int(bbox_str[0])
                ymin = int(bbox_str[1])
                xmax = int(bbox_str[2])
                ymax = int(bbox_str[3])
                save_dict[seq][view].append([fid] + [pid] + [xmin,ymin,xmax-xmin,ymax-ymin] + [1, 0, 0, 0] + feature) # 2048   
            
        np.save('./MGN_mm.npy', save_dict)
    return None
    '''
    
    '''
    # campus version
    save_dict = {}
    from glob import glob
    import json
    
    img_root = '/mnt/sdb/dataset/MOT_datasets/CrossMOT_dataset/CAMPUS/ReID_format/bounding_box_test/'
    file_list = glob('/mnt/sdb/dataset/MOT_datasets/CrossMOT_dataset/CAMPUS/ReID_format/detection_res_test/*.txt')
    file_list.sort()
    seq_list = ['scene1', 'scene2','scene3']
    view_list = ['view1', 'view2', 'view3', 'view4']
    boxes={}
    for file in file_list:
        dets = np.loadtxt(file, delimiter=',').tolist()
        scene = file.split('/')[-1].split('_')[0]
        cam = file.split('/')[-1].split('_')[-1].split('.txt')[0]
        for det in dets:
            img_path = '{}_c{}s{}_f{}.jpg'.format(str(int(det[1])).zfill(4), cam[-1], scene[-1], str(int(det[0])).zfill(6))
            boxes[img_path] = '{},{},{},{}'.format(int(det[2]), int(det[3]), int(det[4]), int(det[5]))

    for seq in seq_list:
        save_dict[seq] = {}
        if seq == 'scene2':
            for view in view_list[:3]:
                save_dict[seq][view] = []
        else:
            for view in view_list:
                save_dict[seq][view] = []             
    
    
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            # Flip test
            if flip_test:
                inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = model(inputs)
                outputs = (outputs + flip_outputs) / 2
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )

            for ind in range(0, len(inputs['images'])):
                feature = outputs[ind].cuda().tolist()
                pid = -1
                img_path = inputs['img_paths'][ind]
                fid = int(img_path.split('/')[-1].split('f')[-1][:6])
                seq = seq_list[int(img_path.split('/')[-1].split('s')[-1].split('_')[0])-1]
                view = view_list[int(img_path.split('/')[-1].split('c')[-1][0]) - 1]
                
                bbox_str = boxes[img_path.split('/')[-1]].split(',')
                
                xmin = int(bbox_str[0])
                ymin = int(bbox_str[1])
                xmax = int(bbox_str[2])
                ymax = int(bbox_str[3])
                save_dict[seq][view].append([fid] + [pid] + [xmin,ymin,xmax-xmin,ymax-ymin] + [1, 0, 0, 0] + feature) # 2048   
            
        np.save('./MGN_campus.npy', save_dict)
    return None
    '''
    '''
    # divo version
    save_dict = {}
    from glob import glob
    import json
    img_root = '/mnt/sdb/dataset/MOT_datasets/CrossMOT_dataset/DIVOTrack/ReID_format/bounding_box_test/'
    seq_list = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
    view_list = ['Drone', 'View1', 'View2']
    f = open('/mnt/sdb/syh/DIVOTrack/datasets/DIVO/boxes.json', 'r')
    boxes = json.load(f)    
    for seq in seq_list:
        save_dict[seq] = {}
        for view in view_list:
            save_dict[seq][view] = []               
    
    
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            # Flip test
            if flip_test:
                inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = model(inputs)
                outputs = (outputs + flip_outputs) / 2
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )

            for ind in range(0, len(inputs['images'])):
                feature = outputs[ind].cuda().tolist()
                pid = -1
                img_path = inputs['img_paths'][ind]
                fid = int(img_path.split('/')[-1].split('f')[-1][:6])
                seq = seq_list[int(img_path.split('/')[-1].split('s')[-1].split('_')[0])-1]
                view = view_list[int(img_path.split('/')[-1].split('c')[-1][0]) - 1]
                
                bbox_str = boxes[img_path.split('/')[-1]].split(',')
                
                xmin = int(bbox_str[0])
                ymin = int(bbox_str[1])
                xmax = int(bbox_str[2])
                ymax = int(bbox_str[3])
                save_dict[seq][view].append([fid] + [pid] + [xmin,ymin,xmax-xmin,ymax-ymin] + [1, 0, 0, 0] + feature) # 2048    
        np.save('./MGN_divo.npy', save_dict)
    return None
    '''
    
    # divo version
    save_dict = {}
    from glob import glob
    import json
    img_root = '/mnt/sdb/syh/DIVOTrack/datasets/DIVO/images/dets/det_imgs/'
    
    view_list = ['View1', 'View2', 'View3']
    seq_list = ['Indoor1', 'Indoor2', 'Outdoor1', 'Outdoor2', 'Park2']
    f = open('/mnt/sdb/syh/DIVOTrack/datasets/DIVO/images/dets/boxes_challenge.json', 'r')
    boxes = json.load(f)    
    for seq in seq_list:
        save_dict[seq] = {}
        for view in view_list:
            save_dict[seq][view] = []               
    
    
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            # Flip test
            if flip_test:
                inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = model(inputs)
                outputs = (outputs + flip_outputs) / 2
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )

            for ind in range(0, len(inputs['images'])):
                feature = outputs[ind].cuda().tolist()
                pid = -1
                img_path = inputs['img_paths'][ind]
                fid = int(img_path.split('/')[-1].split('f')[-1][:6])
                seq = seq_list[int(img_path.split('/')[-1].split('s')[-1].split('_')[0])-1]
                view = view_list[int(img_path.split('/')[-1].split('c')[-1][0]) - 1]
                bbox_str = boxes[img_path.split('/')[-1]]
                
                xmin = int(bbox_str[0])
                ymin = int(bbox_str[1])
                xmax = int(bbox_str[2])
                ymax = int(bbox_str[3])
                save_dict[seq][view].append([fid] + [pid] + [xmin,ymin,xmax-xmin,ymax-ymin] + [1, 0, 0, 0] + feature) # 2048    
        np.save('./MGN_divo_challenge.npy', save_dict)
    return None

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / batch per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / batch per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate()

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
