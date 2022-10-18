# -*- coding: utf-8 -*-
import sys
import os
import glob
import cv2
import numpy as np
import json
#---below---imgaug module
import imgaug as ia
import imgaug.augmenters as iaa
#from imgaug.augmentables import Keypoint, KeypointsOnImage
from PIL import Image 
from labelme import utils
import base64
from shapely import geometry
 
'''
ticks:
1) picture type : png;
2) while augumenting, mask not to go out image shape;
3) maybe some error because data type not correct.
'''
def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.mkdir(path)
        print('====================')
        print('creat path : ', path)
        print('====================')
    return 0
 
 
#-----------------------------Sequential-augument choose here-----
ia.seed(1)
 
 
if __name__ == '__main__':
    # TO-DO-BELOW
    aug_times = 50
    in_dir = "/p9550/LOFAR/LoTSS-DR1/Mingo_png_fr2"
    out_dir = "/p9550/LOFAR/LoTSS-DR1/Mingo_png_fr2_resize"
    #---check-------------
    mkdir(out_dir)
    imgs_dir_list = glob.glob(os.path.join(in_dir, '*.png'))
 
    # for : image
    for idx_jpg_path in imgs_dir_list:
        # get image file
        #idx_img = cv2.imdecode(np.fromfile(idx_jpg_path, dtype=np.uint8), 1)
        img = Image.open(idx_jpg_path)
        # sp = img.size
        idx_img = np.asarray(img)
        #factor = 132.0/idx_img.shape[0]
        #print(factor)
        #seq = iaa.Sequential([
        #    iaa.Affine(
        #           scale=(factor)
        #        )
        # ])
        aug = iaa.Resize({"height": 132, "width": 132}, interpolation="linear")
        img_augmented = aug(image=idx_img)
        #image_aug = seq(image=idx_img)
        print(idx_jpg_path)
        new_img_path = os.path.join(out_dir, idx_jpg_path.split(os.sep)[-1][:-4] + '.png')
        Image.fromarray(img_augmented).save(new_img_path)
