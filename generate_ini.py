import configparser
from glob import glob
import os
import pdb


class myconf(configparser.ConfigParser):
    def __init__(self,defaults=None):
        configparser.ConfigParser.__init__(self,defaults=None)
    def optionxform(self, optionstr):
        return optionstr

data_type = ['train', 'test']
for i in data_type:
    target_path = './datasets/DIVO/images/{}/'.format(i)
    videos = os.listdir(target_path)
    videos.sort()
    for video in videos:
        config=myconf()
        imgs = glob('{}'.format(target_path) + video + '/img1/*.jpg')
        length = len(imgs)
        if 'View1' in video:
            w = 3640
            h = 2048
        else:
            w = 1920
            h = 1080
        config.add_section("Sequence")
        config.set("Sequence","name", video)
        config.set("Sequence","imDir","img1")
        config.set("Sequence","frameRate","30")
        config.set("Sequence","seqLength",'{}'.format(length))
        config.set("Sequence","imWidth",'{}'.format(w))
        config.set("Sequence","imHeight",'{}'.format(h))
        config.set("Sequence","imExt",".jpg")
        o = open(os.path.join('{}'.format(target_path), video, "seqinfo.ini"), "w")
        config.write(o, space_around_delimiters=False)
        o.close()
