'''
Convert the divo to coco format
generate "annotations" directory
'''
import os
import numpy as np
import json
import cv2
import pdb

DATA_PATH = "../../../../datasets/DIVO/images/"
OUT_PATH = DATA_PATH + 'annotations/'
if not os.path.exists(OUT_PATH):
  os.mkdir(OUT_PATH)

SPLITS = ['train', 'test']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True

if __name__ == '__main__':
  for split in SPLITS:
    data_path = DATA_PATH + split
    out_path = OUT_PATH + '{}.json'.format(split)
    out = {'images': [], 'annotations': [], 
           'categories': [],
           'videos': []}
    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for seq in sorted(seqs):
      if '.DS_Store' in seq:
        continue

      video_cnt += 1
      out['videos'].append({
        'id': video_cnt,
        'file_name': seq})
      seq_path = '{}/{}/'.format(data_path, seq)
      img_path = seq_path + 'img1/'
      ann_path = seq_path + 'gt/gt.txt'
      images = sorted(os.listdir(img_path))
      num_images = len([image for image in images if 'jpg' in image])

      image_range = [0, num_images]
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
          # pdb.set_trace()
          frame_id = int(anns[i][0]) - int(anns[:, 0].min())
          if (frame_id < image_range[0] or frame_id > image_range[1]):
            continue
          track_id = int(anns[i][1])
          cat_id = int(anns[i][7])
          ann_cnt += 1
          ann = {'id': ann_cnt,
                 'category_id': 1,
                 'image_id': image_cnt + frame_id,
                 'track_id': track_id,
                 'bbox': anns[i][2:6].tolist(),
                 'conf': 1.}
          out['annotations'].append(ann)
      image_cnt += num_images
    print('loaded {} for {} images and {} samples'.format(
      split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
        
      
