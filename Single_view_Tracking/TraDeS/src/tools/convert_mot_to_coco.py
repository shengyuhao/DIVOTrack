import os
from unicodedata import category
import numpy as np
import json
import cv2

# Use the same script for MOT16
# DATA_PATH = '../../data/mot16/'
DATA_PATH = '../../data/mot17/'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['train', 'test'] # 'train_half', 'val_half', 'train', 'test'
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = False

if __name__ == '__main__':
  for split in SPLITS:
    data_path = DATA_PATH + split
    print(data_path)
    out_path = OUT_PATH + '{}.json'.format(split)
    out = {'images': [], 'annotations': [],
           'categories': [],
           'videos': []}
    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    global_track_id = {}
    for seq in sorted(seqs):
      if '.DS_Store' in seq:
        continue
      # if 'mot17' in DATA_PATH and (split != 'test' and not ('FRCNN' in seq)):
      #   continue
      video_cnt += 1
      out['videos'].append({
        'id': video_cnt,
        'file_name': seq})
      seq_path = '{}/{}/'.format(data_path, seq)
      img_path = seq_path + 'img1/'
      ann_path = seq_path + 'gt/gt.txt'
      images = os.listdir(img_path)
      images.sort()
      num_images = len([image for image in images if 'jpg' in image])

      if split == 'train':
        image_range = [0, num_images] 
      else:
        image_range = [0, int(num_images / 2)] 
      print(num_images,image_range)
      for i in range(num_images):
        if (i < image_range[0] or i > image_range[1]):
          continue
        image_info = {'file_name': '{}/img1/{}'.format(seq, images[i-image_range[0]]),
                      'id': image_cnt + i + 1,
                      'frame_id': i + 1 - image_range[0],
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id': \
                        image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt}
        out['images'].append(image_info)
      print('{}: {} images'.format(seq, num_images))
      if split != 'test':
        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        print(' {} ann images'.format(int(anns[:, 0].max())))
        for i in range(anns.shape[0]):
          frame_id = int(anns[i][0] - int(anns[:, 0].min()))
          if (frame_id < image_range[0] or frame_id > image_range[1]):
            continue
          track_id = int(anns[i][1])
          cat_id = int(anns[i][7])
          ann_cnt += 1

          category_id = 1
          identity = '{}_{}'.format(video_cnt,track_id)
          if identity not in global_track_id:
            if global_track_id:
              global_track_id.update({identity:global_track_id[list(global_track_id)[-1]] + 1})
            else:
              global_track_id.update({identity: 1})
          ann = {'id': ann_cnt,
                 'category_id': category_id,
                 'image_id': image_cnt + frame_id,
                 'track_id': track_id,
                 'bbox': anns[i][2:6].tolist(),
                 'conf': float(1),
                 'global_track_id':global_track_id[identity],
                 'iscrowd': 0}
          #  ann['bbox'] x1 y2 w h
          ann['area'] = ann['bbox'][2] * ann['bbox'][3]
          out['annotations'].append(ann)
      image_cnt += num_images
    print('total # identities:', len(list(global_track_id)))
    print('loaded {} for {} images and {} samples'.format(
      split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
