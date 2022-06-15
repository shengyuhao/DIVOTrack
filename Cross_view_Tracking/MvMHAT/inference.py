from __future__ import division, print_function, absolute_import
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from loader import Loader
import torchvision.models as models
from deep_sort.mvtracker import MVTracker
from deep_sort.update import Update
from torch.cuda.amp import autocast as autocast
import config as C
from collections import defaultdict
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=C.INF_ID)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = C.TRAIN_GPUS
resume = "./models/" + args.model + '.pth'


def read_loader(dataset_name):
    dataset = Loader(frames=1, views=4, mode='test', dataset=dataset_name)
    dataset_loader = DataLoader(dataset, num_workers=4, pin_memory=True)
    dataset_info = {
        'view': dataset.view_ls,
        'seq_len': len(dataset),
        'start': dataset.cut_dict[dataset_name][0],
        'end': dataset.cut_dict[dataset_name][1]
    }
    return dataset_info, dataset_loader

def search_bbox(index, id, det_list):
    for item in det_list:
        if index == item[0] and id == item[1]:
            return item[2:6]
    return []

def gather_seq_info_multi_view(dataset_name, dataset_info, dataset_test, model):

    groundtruth = None
    seq_dict = {}
    coffidence = 1
    print('loading dataset...')


    feature_dict = defaultdict(list)

    image_filenames = defaultdict(list)
    detections = defaultdict(list)
    for data_i, data in tqdm(enumerate(dataset_test), total=len(dataset_test)):
        for view_i, view in enumerate(dataset_info['view']):
            data_pack = data[view_i][0]
            if data_pack == []:
                continue
            img, box, lbl, scn = data[view_i][0]
            model.eval()
            with torch.no_grad():
                img = img.squeeze(0).cuda()
                with autocast():
                    img = model(img)
            image_filenames[view].append(scn[0])
            for feature, bndbox, id in zip(img, box, lbl):
                index = int(scn[0].split('/')[-1].split('_')[-1].split('.')[0])
                bndbox = [int(i) for i in bndbox]
                id = int(id[0])
                det = [index] + [id] + bndbox + [coffidence] + [0, 0, 0] + feature.detach().cpu().numpy().tolist()
                fea = [index] + [id] + bndbox + [coffidence] + [0, 0, 0] + feature.detach().cpu().numpy().tolist()
                detections[view].append(det)
                feature_dict[view].append(fea)

    # for view_i, view in enumerate(dataset_info['view']):
    #     seq_dict[view] = {
    #     "sequence_name": 'test',
    #     "image_filenames": image_filenames[view],
    #     "detections": np.array(detections[view]),
    #     "groundtruth": groundtruth,
    #     "image_size": (3, 1080, 1920),
    #     "min_frame_idx": dataset_info['start'],
    #     "max_frame_idx": dataset_info['end'] - 1,
    #     "feature_dim": 1000,
    #     "update_ms": 10
    #     }
    return feature_dict


def run(dataset_name, display, dataset, model):
    dataset_info, dataset_loader = read_loader(dataset)
    feature_dict = gather_seq_info_multi_view(dataset_name, dataset_info, dataset_loader, model)
    return feature_dict

    # mvtracker = MVTracker(dataset_info['view'])

    # updater = Update(seq=seq_mv, mvtracker=mvtracker, display=display)
    # updater.run()
    # for view in updater.view_ls:
    #     if not os.path.exists(output_file):
    #         os.makedirs(output_file)
    #     f = open(output_file + dataset + '_' + view + '.txt', 'w')
    #     for row in updater.result[view]:
    #         print('%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1,-1' % (
    #             row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    #     f.close()

if __name__ == "__main__":
    model = models.resnet50(pretrained=False)
    model = model.cuda()

    save_feature = defaultdict(list)
    print('model: ' + args.model)
    if resume:
        checkpoint_path = resume
        ckp = torch.load(checkpoint_path)['model']
        model.load_state_dict(ckp)
    else:
        checkpoint_path = './models/pretrained.pth'
        ckp = torch.load(checkpoint_path)
        model.load_state_dict(ckp)
    for dataset_name in C.TEST_DATASET:
        save_feature[dataset_name] = run(dataset_name , display=C.DISPLAY, dataset=dataset_name, model=model)

    np.save('./feature/self_0603_feature.npy', save_feature)
