import json
import sys

import cv2
import os
from PIL import Image
import math
import numpy as np
from glob import glob
from numpy import random
import colorsys

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)

# def img2video(im_dir, video_dir, video_name, file_inpath):
#     f = open("{}/new_{}.txt".format(file_inpath, video_name), "r")
#     # colors = [(255 - random.randint(20) * i, 255 - random.randint(20) * i, 255 - random.randint(20) * i) for i in
#     # range(10)]
#     colors = []
#     for i1 in range(5):
#         for i2 in range(5):
#             colors.append((random.randint(255), random.randint(255), random.randint(255)))
#     b = [i.split() for i in f.readlines()]
#     if not os.path.exists(video_dir):
#         os.makedirs(video_dir)
#     # set saved fps
#     fps = 3
#     # get frames list
#     frame_path = glob(im_dir + "/*jpg")
#     frame_path.sort()
#     frames = [file for file in frame_path if video_name in file]
#     # frames = sorted(os.listdir(im_dir))
#     # w,h of image
#     img = cv2.imread(frames[0])
#     img_size = (img.shape[1], img.shape[0])
#     # splice video_dir
#     video_dir = os.path.join(video_dir, video_name + '.mp4')
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     # also can write like:fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     # if want to write .mp4 file, use 'MP4V'
#     videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
#
#     for frame in frames:
#         # f_path = os.path.join(im_dir, frame)
#         image = cv2.imread(frame)
#
#         for bb in b:
#             # pdb.set_trace()
#             if str(int(float(bb[0]))).zfill(6) == frame.split("_")[-1].split(".jpg")[0]:
#                 try:
#                     pt1 = (int(float(bb[2])), int(float(bb[3])))
#                     pt2 = (int(float(bb[4])), int(float(bb[5])))
#                     # Blue color in BGR
#                     # color = (255, 0, 0)
#                     # Line thickness of 2 px
#                     thickness = 2
#                     image = cv2.rectangle(image, pt1, pt2, colors[int(bb[1]) % 25 + int(bb[1]) % 10], thickness)
#                     cv2.putText(image, '{}'.format(bb[1]), (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
#                                 colors[int(bb[1]) % 25 + + int(bb[1]) % 10], 2)
#                 except:
#                     print(frame)
#         videowriter.write(image)
#         # print(frame + " has been written!")
#
#     videowriter.release()


def create_avi(file_inpath, video_inpath, filename):
    f = open("{}/{}.txt".format(file_inpath, filename), "r")
    print("{}/{}.mp4".format(video_inpath, filename))
    cap = cv2.VideoCapture("{}/{}.mp4".format(video_inpath, filename))
    # f = open("origin_MOT_files/circleRegion_Drone.txt", "r")
    # cap = cv2.VideoCapture("Data/videosrc/circleRegion/circleRegion_Drone.MP4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter('{}/{}.avi'.format(file_inpath, filename), fourcc, fps, size)
    # videoWriter = cv2.VideoWriter('test.avi', fourcc, fps, size)
    b = [i.split() for i in f.readlines()]
    pre_frame = -1
    frame = None
    for bb in b:
        print(bb)
        try:
            r, g, b = create_unique_color_uchar(int(float(bb[1])) + 10)
            pt1 = (int(float(bb[2])), int(float(bb[3])))
            pt2 = (int(float(bb[4])), int(float(bb[5])))
            cur_frame = int(bb[0])
            if cur_frame != pre_frame:
                if pre_frame != -1:
                    videoWriter.write(frame)
                    # print(frame.shape)
                    # print out current image
                    # filename = 'savedImage{bb[1]}.jpg'
                    # cv2.imwrite(filename, frame)
                    # break
                pre_frame = cur_frame
                ret, frame = cap.read()
                # print(ret)

            # Blue color in BGR
            color = (255, 0, 0)

            # Line thickness of 2 px
            thickness = 2

            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            frame = cv2.rectangle(frame, pt1, pt2, (r,g,b), thickness)
            cv2.putText(frame, '{}'.format(bb[1]), (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (r,g,b), 2)
        except:
            pass

        # Displaying the image
    videoWriter.write(frame)

    # Using cv2.imwrite() method
    # Saving the image

    # frame = cv2.rectangle(frame,pt1,pt2,(255, 0, 0) )


def save_frame(file_path, img_path, output_path, flag):
    # colors = [(255 - 50 * i, 255 - 50 * i, 255 - 50 * i) for i in range(20)]
    f = open("{}".format(file_path), "r")
    b = [i.split(',') for i in f.readlines()]
    img = cv2.imread(img_path)
    ind = 0
    thickness = 10
    for bb in b:
        if int(float(bb[0])) != flag+1:
            continue
        
        r, g, b = create_unique_color_uchar(int(float(bb[1]))+10)
        pt1 = (int(float(bb[2])), int(float(bb[3])))
        pt2 = (int(float(bb[4])+float(bb[2])), int(float(bb[5])+float(bb[3])))
        frame = cv2.rectangle(img, pt1, pt2, (r, g, b), thickness)
        cv2.putText(frame, '{}'.format(bb[1]), (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (r, g, b), thickness)
        ind += 1
    img = cv2.resize(img, (1920, 1080))
    cv2.imwrite(output_path, img)
    print(ind)

def img2video(im_dir, video_path):

    fps = 30
    # get frames list
    frames = glob(im_dir + "/*jpg")
    frames.sort()
    # frames = sorted(os.listdir(im_dir))
    # w,h of image
    img = cv2.imread(frames[0])
    img_size = (img.shape[1], img.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)

    for frame_id, frame in enumerate(frames):
        if frame_id > -1:
            image = cv2.imread(frame)
            videowriter.write(image)

    videowriter.release()

if __name__ == "__main__":
    path = "/home/syh/shengyuhao/tracking_wo_bnw/output/tracktor/MOT17/Tracktor++/circleRegion_Drone.txt"
    img_path = '/data/syh/datasets/STREET/images/test/circleRegion_Drone/img1/'
    img_list = glob(img_path + '/*.jpg')
    img_list.sort()
    for ind, img in enumerate(img_list):
        if ind > 100:
            break
        output_path = '/home/syh/shengyuhao/tracking_wo_bnw/output/tracktor/MOT17/images/circle_drone/{}.jpg'.format(str(ind).zfill(6))
        file_path = path
        save_frame(file_path, img, output_path, ind)
    # # name = ["southGate", "park", "shopFrontGate", "shopSideSquare", "shopSideGate", "playground", "circleRegion",
    # #         "shopSecondFloor", "movingView", "innerShop"]
    # name = ["shopSideGate"]
    # file_type = 'test_gt'
    # view = ["Drone", "View1", "View2"]
    # for i in name:
    #     for j in view:
    #         img_path = "E:/Share/comparedVideo/MultiViewPairs/matchedFramesrc/{}/{}_{}_000090.jpg".format(i, i, j)
    #         output_path = './data_statistic/imgs/{}_{}_{}_1.jpg'.format(file_type, i, j)
    #         file_path = path + file_type + '/{}/{}.txt'.format(i, j)
    #         save_frame(file_path, img_path, output_path, j)
    #         # video_path = "./data/0716"
    #         # img2video(img_path, video_path, i + "_" + j, "./data/0716/unmodified")
    #     # create_avi("./data/0710", "./test/"+i, i + "_" + j)
    # # create_avi("splited_MOT_files", "Data/videosrc/park", "park_View1")
    #img_path = r'C:\Users\Admin\Desktop\New folder1\demos\demos\MOT17-07-SDP_frame_best015'
    #video_path = r'C:\Users\Admin\Desktop\New folder1\demos\demos\MOT17-07.mp4'
    #img2video(img_path, video_path)
