from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from collections import defaultdict
from ..generic_dataset import GenericDataset

class DIVO(GenericDataset):
  num_categories = 1
  default_resolution = [544, 960]
  class_name = ['']
  max_objs = 256
  cat_ids = {1: 1, -1: -1}
  def __init__(self, opt, split):

    print('Using DIVO {}'.format(split))

    data_dir = os.path.join(opt.data_dir, 'images')
    
    ann_file = '{}.json'.format('train' if split == 'train' else 'test')

    img_dir = os.path.join(data_dir, '{}'.format('train' if split == 'train' else 'test'))
    print('ann_file', ann_file)
    ann_path = os.path.join(data_dir, 'annotations', ann_file)

    self.images = None
    # load image list and coco
    super(DIVO, self).__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)
    print('Loaded DIVO {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    results_dir = os.path.join(save_dir, 'results_divo')
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)
    for video in self.coco.dataset['videos']:
      video_id = video['id']
      file_name = video['file_name']
      out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
      f = open(out_path, 'w')
      images = self.video_to_images[video_id]
      tracks = defaultdict(list)
      for image_info in images:
        if not (image_info['id'] in results):
          continue
        result = results[image_info['id']]
        frame_id = image_info['frame_id']
        for item in result:
          if not ('tracking_id' in item):
            item['tracking_id'] = np.random.randint(100000)
          if item['active'] == 0:
            continue
          tracking_id = item['tracking_id']
          bbox = item['bbox']
          bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
          tracks[tracking_id].append([frame_id] + bbox)
      rename_track_id = 0
      for track_id in sorted(tracks):
        rename_track_id += 1
        for t in tracks[track_id]:
          f.write('{},{},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1,-1\n'.format(
            t[0], rename_track_id, t[1], t[2], t[3]-t[1], t[4]-t[2]))
      f.close()
  
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)

    os.system('python tools/eval_motchallenge.py ' + \
              '../../../datasets/DIVO/images/{}/ '.format('test') + \
              '{}/results_divo/' + ' --eval_official')
