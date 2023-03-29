from cProfile import label
import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import json
import numpy as np
import torch
import copy

from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from cython_bbox import bbox_overlaps as bbox_ious
from opts import opts
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta
import utils


class LoadImages_DIVO:  # for inference
    def __init__(self, opt, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = [".jpg", ".jpeg", ".png", ".tif"]
            self.files = sorted(glob.glob("%s/*.*" % path))
            self.files = list(
                filter(
                    lambda x: os.path.splitext(x)[1].lower() in image_format, self.files
                )
            )
        elif os.path.isfile(path):
            self.files = [path]
        seq_info, seq_length = None, 0
        for filename in os.listdir(path):
            if filename.split(".")[-1] == "ini":
                seq_info = open(osp.join(path, filename)).read()

        if seq_info != None:
            seq_length = int(
                seq_info[seq_info.find("seqLength=") + 10 : seq_info.find("\nimWidth")]
            )
        file_list = []
        self.view_list = []
        for filename in self.files:
            name = filename.split(".")[0]
            # gather the view
            if name.split("_")[-2] not in self.view_list:
                self.view_list.append(name.split("_")[-2])
            if opt.test_divo:
                file_list.append(filename)
                seq_length = (
                    int(name.split("_")[-1])
                    if int(name.split("_")[-1]) > seq_length
                    else seq_length
                )
            if opt.test_mvmhat or opt.test_mvmhat_campus or opt.test_wildtrack:
                if int(name.split("_")[-1]) > int(seq_length * 2 / 3):
                    file_list.append(filename)
            if opt.test_epfl:
                if int(name.split("_")[-1]) >= int(seq_length):
                    file_list.append(filename)
        self.view_list.sort()
        self.files = file_list
        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0
        self.seq_length = seq_length
        assert self.nF > 0, "No images found in " + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        img0 = cv2.resize(img0, (1920, 1080))
        assert img0 is not None, "Failed to load " + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        img0 = cv2.resize(img0, (1920, 1080))
        assert img0 is not None, "Failed to load " + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadImages:  # for inference
    def __init__(self, opt, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = [".jpg", ".jpeg", ".png", ".tif"]
            self.files = sorted(glob.glob("%s/*.*" % path))
            self.files = list(
                filter(
                    lambda x: os.path.splitext(x)[1].lower() in image_format, self.files
                )
            )
        elif os.path.isfile(path):
            self.files = [path]
        for filename in os.listdir(path):
            if filename.split(".")[-1] == "ini":
                seq_info = open(osp.join(path, filename)).read()
        seq_length = int(
            seq_info[seq_info.find("seqLength=") + 10 : seq_info.find("\nimWidth")]
        )

        file_list = []
        self.view_list = []
        for filename in self.files:
            name = filename.split(".")[0]
            # gather the view
            if name.split("_")[-2] not in self.view_list:
                self.view_list.append(name.split("_")[-2])
            if opt.test_divo:
                if int(name.split("_")[-1]) <= int(seq_length):
                    file_list.append(filename)
            if opt.test_mvmhat or opt.test_mvmhat_campus or opt.test_wildtrack:
                if int(name.split("_")[-1]) > int(seq_length * 2 / 3):
                    file_list.append(filename)
            if opt.test_epfl:
                if int(name.split("_")[-1]) >= int(seq_length):
                    file_list.append(filename)
        self.view_list.sort()
        self.files = file_list
        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0
        assert self.nF > 0, "No images found in " + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        img0 = cv2.resize(img0, (1920, 1080))
        assert img0 is not None, "Failed to load " + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        img0 = cv2.resize(img0, (1920, 1080))
        assert img0 is not None, "Failed to load " + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadImagesAndLabels:  # for training
    def __init__(self, path, img_size=(1088, 608), augment=False, transforms=None):
        with open(path, "r") as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace("\n", "") for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [
            x.replace("images", "labels_with_ids")
            .replace(".png", ".txt")
            .replace(".jpg", ".txt")
            for x in self.img_files
        ]

        self.nF = len(self.img_files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path, zero_start=False):
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        rs = 0
        if (img.shape[0], img.shape[1]) != (1080, 1920):
            img = cv2.resize(img, (1920, 1080))
        if img is None:
            raise ValueError("File corrupt {}".format(img_path))
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)
        # if label_path.split('/')[-1].split('_')[1] == 'View1' and label_path.split('/')[-1].split('_')[0] == 'southGate':
        #     cv2.imwrite('/mnt/sdb/dataset/MOT_datasets/lpy/test1.png', img)
        # Load labels
        if os.path.isfile(label_path) and os.path.getsize(label_path) != 0:
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 1] = labels[:, 1] + 1 if zero_start else labels[:, 1]
            labels[:, 2] = ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            labels[:, 3] = ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            labels[:, 4] = ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            labels[:, 5] = ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
        else:
            labels = np.array([])

        # Augment image and labels
        if self.augment:
            img, labels, M = random_affine(
                img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20)
            )

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


def letterbox(
    img, height=608, width=1088, color=(127.5, 127.5, 127.5)
):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (
        round(shape[1] * ratio),
        round(shape[0] * ratio),
    )  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # padded rectangular
    return img, ratio, dw, dh


def random_affine(
    img,
    targets=None,
    degrees=(-10, 10),
    translate=(0.1, 0.1),
    scale=(0.9, 1.1),
    shear=(-2, 2),
    borderValue=(127.5, 127.5, 127.5),
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(
        angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s
    )

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[
        0
    ] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[
        1
    ] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(
        (random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180
    )  # x shear (deg)
    S[1, 0] = math.tan(
        (random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180
    )  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(
        img, M, dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=borderValue
    )  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = (
                np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            )

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = (
                np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
                .reshape(4, n)
                .T
            )

            # reject warped points outside of image
            # np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            # np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            # np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            # np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]
            targets = targets[targets[:, 2] < width]
            targets = targets[targets[:, 4] > 0]
            targets = targets[targets[:, 3] < height]
            targets = targets[targets[:, 5] > 0]

        return imw, targets, M
    else:
        return imw


def collate_fn(batch):
    imgs, labels, paths, sizes = zip(*batch)
    batch_size = len(labels)
    imgs = torch.stack(imgs, 0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [torch.from_numpy(l) for l in labels]
    filled_labels = torch.zeros(batch_size, max_box_len, 6)
    labels_len = torch.zeros(batch_size)

    for i in range(batch_size):
        isize = labels[i].shape[0]
        if len(labels[i]) > 0:
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize

    return imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1)


class JointDataset(LoadImagesAndLabels):  # for training
    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(
        self, opt, root, paths, img_size=(1088, 608), augment=False, transforms=None
    ):
        self.opt = opt
        self.baseline = self.opt.baseline
        self.baseline_view = self.opt.baseline_view
        self.single_view_id_split_loss = self.opt.single_view_id_split_loss
        self.cross_view_id_split_loss = self.opt.cross_view_id_split_loss
        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1
        self.zero_start = self.opt.zero_start

        for ds, path in paths.items():
            with open(path, "r") as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [
                    osp.join(root, x.strip()) for x in self.img_files[ds]
                ]
                self.img_files[ds] = list(
                    filter(lambda x: len(x) > 0, self.img_files[ds])
                )

            self.label_files[ds] = [
                x.replace("images", "labels_with_ids")
                .replace(".png", ".txt")
                .replace(".jpg", ".txt")
                for x in self.img_files[ds]
            ]

        self.sce = []
        self.view = []
        for ds, label_paths in self.label_files.items():
            for lp in label_paths:
                scene = (lp.split("/")[-1]).split("_")[0]
                view = (lp.split("/")[-1]).split("_")[1]
                if scene not in self.sce:
                    self.sce.append(scene)
                if view not in self.view:
                    self.view.append(view)
        self.sce.sort()
        self.view.sort()

        self.id_array = [-1 for i in range(len(self.sce))]
        self.single_view_id_array = [
            [-1 for i in range(len(self.view))] for i in range(len(self.sce))
        ]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            view_max_index = -1
            for lp in label_paths:
                scene = (lp.split("/")[-1]).split("_")[0]
                view = (lp.split("/")[-1]).split("_")[1]
                if self.id_array[self.sce.index(scene)] == -1:
                    max_index = -1
                if (
                    self.single_view_id_array[self.sce.index(scene)][
                        self.view.index(view)
                    ]
                    == -1
                ):
                    view_max_index = -1
                if osp.getsize(lp) != 0:
                    lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
                if img_max > view_max_index:
                    view_max_index = img_max
                self.id_array[self.sce.index(scene)] = (
                    max_index + 1 if self.zero_start else max_index
                )
                self.single_view_id_array[self.sce.index(scene)][
                    self.view.index(view)
                ] = (view_max_index + 1 if self.zero_start else view_max_index)
            self.tid_num[ds] = sum(self.id_array)

        # remove any non exist view
        for l in self.single_view_id_array:
            for i in l:
                if i == -1:
                    l.remove(i)

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v
        self.nID = int(sum(self.id_array))
        self.view_nID = int(sum(sum(i) for i in self.single_view_id_array))
        # modified loss array
        self.single_loss_array = [0]

        for i in range(len(self.single_view_id_array)):
            for j in range(len(self.single_view_id_array[i])):
                self.single_loss_array.append(
                    int(self.single_loss_array[-1] + self.single_view_id_array[i][j])
                )
        opt.single_loss_array = self.single_loss_array

        self.cross_loss_array = [0]
        for i in range(len(self.id_array)):
            self.cross_loss_array.append(
                int(self.cross_loss_array[-1] + self.id_array[i])
            )
        opt.cross_loss_array = self.cross_loss_array

        # self.nds = total length of the dataset = [25299]
        self.nds = [len(x) for x in self.img_files.values()]
        # self.cds = 0
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        # self.nF = 25299
        self.nF = sum(self.nds)
        # width = 1088, height = 608
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms
        print("=" * 80)
        print("dataset summary")
        print(self.tid_num)
        print("total # identities:", self.nID)
        print("start index")
        print(self.tid_start_index)
        print("=" * 80)

    def __getitem__(self, files_index):
        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        imgs, labels, img_path, (input_h, input_w) = self.get_data(
            img_path, label_path, self.zero_start
        )

        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        scene = (img_path.split("/")[-1]).split("_")[0]
        view = (img_path.split("/")[-1]).split("_")[1]
        # 1088 608 / 4
        output_h = imgs.shape[1] // self.opt.down_ratio
        output_w = imgs.shape[2] // self.opt.down_ratio

        num_classes = self.num_classes
        num_objs = labels.shape[0]
        # (1, 152, 272)
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        if self.opt.ltrb:
            wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        else:
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs,), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs,), dtype=np.uint8)
        ids = np.zeros((self.max_objs,), dtype=np.int64)
        single_view_ids = np.zeros((self.max_objs,), dtype=np.int64)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
        for k in range(min(num_objs, self.max_objs)):
            label = labels[k]
            # normalized bbox (lx, ly, w, h)
            bbox = label[2:]
            # cls_id = 0
            cls_id = int(label[0])
            # bbox (lx, ly, w, h) * 272 152
            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            # copy of bbox
            bbox_amodal = copy.deepcopy(bbox)
            # offset bbox
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.0
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.0
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
            # adjust the ordinate value
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]
            # offset bbox
            bbox_xy = copy.deepcopy(bbox)
            bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
            bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
            bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
            bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = 6 if self.opt.mse_loss else radius
                # radius = max(1, int(radius)) if self.opt.mse_loss else radius
                ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                if self.opt.ltrb:
                    wh[k] = (
                        ct[0] - bbox_amodal[0],
                        ct[1] - bbox_amodal[1],
                        bbox_amodal[2] - ct[0],
                        bbox_amodal[3] - ct[1],
                    )
                else:
                    wh[k] = 1.0 * w, 1.0 * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = label[1] + sum(
                    [self.id_array[i] for i in range(self.sce.index(scene))]
                )
                ids[k] = ids[k] if self.cross_view_id_split_loss else ids[k] - 1
                s_index = (
                    self.view.index(view)
                    + sum(
                        len(self.single_view_id_array[i])
                        for i in range(self.sce.index(scene))
                    )
                    if self.sce.index(scene) != 0
                    else self.view.index(view)
                )
                single_view_ids[k] = self.single_loss_array[s_index] + label[1]
                single_view_ids[k] = (
                    single_view_ids[k]
                    if self.single_view_id_split_loss
                    else single_view_ids[k] - 1
                )
                bbox_xys[k] = bbox_xy
        if self.baseline == 0:
            ret = {
                "input": imgs,
                "hm": hm,
                "reg_mask": reg_mask,
                "ind": ind,
                "wh": wh,
                "reg": reg,
                "ids": ids,
                "single_view_ids": single_view_ids,
                "bbox": bbox_xys,
            }
        else:
            if self.baseline_view == 0:  # single view id
                ret = {
                    "input": imgs,
                    "hm": hm,
                    "reg_mask": reg_mask,
                    "ind": ind,
                    "wh": wh,
                    "reg": reg,
                    "single_view_ids": single_view_ids,
                    "bbox": bbox_xys,
                }
            else:  # cross view id
                ret = {
                    "input": imgs,
                    "hm": hm,
                    "reg_mask": reg_mask,
                    "ind": ind,
                    "wh": wh,
                    "reg": reg,
                    "ids": ids,
                    "bbox": bbox_xys,
                }
        return ret
