from torch.utils.data import Dataset
from collections import defaultdict
import os
import numpy as np
import cv2
import random
import re
import config as C
from glob import glob

class Loader(Dataset):
    def __init__(self, views=3, frames=2, mode='train', dataset='1'):
        self.video_name = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
        # self.video_name = ['park']
        self.views_name = ['Drone', 'View1', 'View2']
        self.views = views
        self.mode = mode
        self.dataset = dataset
        self.down_sample = 1
        self.root_dir = os.path.join(C.ROOT_DIR)
        self.img_root = os.path.join(C.ROOT_DIR, 'images')
        self.dataset_dir = os.path.join(self.img_root, dataset)

        if self.mode == 'train':
            self.cut_dict = {
                'circleRegion': [1601, 3200],
                'innerShop': [1101, 2200],
                'movingView': [531, 1160],  
                'park': [601, 1200],
                'playground': [901, 1800],
                'shopFrontGate': [1251, 2500],
                'shopSecondFloor': [826, 1650],
                'shopSideGate': [751, 1500],
                'shopSideSquare': [601, 1200],
                'southGate': [801, 1600]
            }
        else:
            self.cut_dict = {
                'circleRegion': [0, 1600],
                'innerShop': [0, 1100],
                'movingView': [0, 580],
                'park': [0, 600],
                'playground': [0, 900],
                'shopFrontGate': [0, 1250],
                'shopSecondFloor': [0, 825],
                'shopSideGate': [0, 750],
                'shopSideSquare': [0, 600],
                'southGate': [0, 800]
            }
        if self.mode == 'train':
            self.frames = frames
            self.isShuffle = C.DATASET_SHUFFLE
            self.isCut = 1
        elif self.mode == 'test':
            self.frames = 1
            self.isShuffle = 0
            self.isCut = 1


        self.view_ls = self.views_name
        self.img_dict = self.gen_path_dict(False)
        self.anno_dict, self.max_id, self.view_id_ls = self.gen_anno_dict()

    def gen_path_dict(self, drop_last: bool):
        path_dict = defaultdict(list)
        frames_all = glob(self.dataset_dir + "/*.jpg")
        for view in self.view_ls:
            # dir = os.path.join(self.dataset_dir, view, 'images')
            # path_ls = os.listdir(dir)
            
            # # path_ls.sort(key=lambda x: int(x[:-4]))
            # path_ls.sort(key=lambda x: int(re.search(r"\d*", x).group()))
            # path_ls = [os.path.join(dir, i) for i in path_ls]
            path_ls = [img_path for img_path in frames_all if view in img_path]
            path_ls.sort()
            if self.isCut:
                start, end = self.cut_dict[self.dataset][0], self.cut_dict[self.dataset][1]
                path_ls = path_ls[start:end]
            if drop_last:
                path_ls = path_ls[:-1]
            cut = len(path_ls) % self.frames
            if cut:
                path_ls = path_ls[:-cut]
            if self.isShuffle:
                random.seed(self.isShuffle)
                random.shuffle(path_ls)
            path_dict[view] += path_ls
        path_dict = {view: [path_dict[view][i:i + self.frames] for i in range(0, len(path_dict[view]), self.frames)] for
                     view in path_dict}
        return path_dict

    def gen_anno_dict(self):
        anno_dict = {}
        max_id = -1
        view_maxid = -1
        view_id_ls = []

        for view in self.view_ls:
            anno_view_dict = defaultdict(list)
            if self.mode == 'train':
                anno_path = os.path.join(self.root_dir, 'train_gt', self.dataset, view + '.txt')
            elif self.mode == 'test':
                anno_path = os.path.join(C.DETECTION_DIR, '{}_{}.txt'.format(self.dataset, view))
            with open(anno_path, 'r') as anno_file:
                anno_lines = anno_file.readlines()
                for anno_line in anno_lines:
                    if self.mode == 'train':
                        anno_line_ls = anno_line.split(' ')
                    else:
                        anno_line_ls = anno_line.split(',')
                    anno_key = str(int(anno_line_ls[0]))
                    anno_view_dict[anno_key].append(anno_line_ls)
                    if max_id < int(anno_line_ls[1]):
                        max_id = int(anno_line_ls[1])
                    if view_maxid < int(anno_line_ls[1]):
                        view_maxid = int(anno_line_ls[1])
            view_id_ls.append(view_maxid)
            view_maxid = -1
            anno_dict[view] = anno_view_dict
        return anno_dict, max_id, view_id_ls

    def read_anno(self, path: str):
        path_split = path.split('/')
        view = path_split[-1].split('.txt')[0].split("_")[1]
        # frame = str(int(re.search(r"\d*", path_split[-1]).group()))
        frame = path_split[-1].split('.jpg')[0].split("_")[-1]
        annos = self.anno_dict[view][str(int(frame))]
        bbox_dict = {}
        for anno in annos:
            bbox = anno[2:6]
            # bbox = [int(float(i)) for i in bbox]
            xmin = int(float(bbox[0]))
            ymin = int(float(bbox[1]))
            xmax = int(float(bbox[2]))
            ymax = int(float(bbox[3]))
            bbox_trans = [xmin, ymin, xmax-xmin, ymax-ymin]
            if xmax - xmin <= 0 or ymax - ymin <= 0:
                continue
            bbox_dict[anno[1]] = bbox_trans
        return bbox_dict

    def crop_img(self, frame_img, bbox_dict):
        img = cv2.imread(frame_img)
        c_img_ls = []
        bbox_ls = []
        label_ls = []
        for key in bbox_dict:
            bbox = bbox_dict[key]
            bbox = [0 if i < 0 else i for i in bbox]
            # c_img_ls.append(img[bbox[0]:bbox[2], bbox[1]:bbox[3], :])
            crop = img[bbox[1]:bbox[3] + bbox[1], bbox[0]:bbox[2] + bbox[0], :]
            crop = cv2.resize(crop, (224, 224)).transpose(2, 0, 1).astype(np.float32)
            c_img_ls.append(crop)
            bbox_ls.append(bbox)
            label_ls.append(key)
        return np.stack(c_img_ls), bbox_ls, label_ls, frame_img

    def __len__(self):
        # return self.len
        return min([len(self.img_dict[i]) for i in self.view_ls] + [10000])

    def __getitem__(self, item):
        ret = []
        img_ls = [self.img_dict[view][item] for view in self.view_ls]

        for img_view in img_ls:
            view_ls = []
            for img in img_view:
                anno = self.read_anno(img)
                if anno == {}:
                    if self.mode == 'train':
                        return self.__getitem__(item - 1)
                    else:
                        view_ls.append([])
                        continue
                view_ls.append(self.crop_img(img, anno))
            ret.append(view_ls)
        return ret



if __name__ == '__main__':
    a = Loader(mode='train', dataset='1')
    for i in enumerate(a):
        pass






