import json

import argparse

import os
import sys
import random
import itertools
import colorsys

import skimage.io
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
#from matplotlib import patches,  lines
#from matplotlib.patches import Polygon
import pycocotools.mask as cocomask

import visualize
import pandas

import cv2

parser = argparse.ArgumentParser(description='plot v3 mask results')

parser.add_argument('--resultsfile', dest='results_file', type=str, default='/home/blao/RGC-Mask-Transfiner/lofar_test.csv', help='object detect results json file')
parser.add_argument('--imagedir', dest='image_dir', type=str, default='/home/blao/RGC-Mask-Transfiner/datasets/coco/LOFAR', help='input image directory')
parser.add_argument('--outdir', dest='out_dir', type=str, default='/home/blao/RGC-Mask-Transfiner/vis_hetu_lofar', help='output directory')

args = parser.parse_args()

results = args.results_file
image_dir = args.image_dir
out_dir = args.out_dir

cls = ['cs','fr1','fr2','ht','cj']

csv_hetu = pandas.read_csv(results)
imagefiles = csv_hetu['imagefilename'].values
labels = csv_hetu['label'].values
scores = csv_hetu['score'].values
boxs = csv_hetu['box'].values
masks = csv_hetu['mask'].values

file_nms = os.listdir(image_dir)

for fn in file_nms:
    if not fn.endswith('.png'):
        continue
    png_fn = fn.replace('.png', '_pred.png')
    if os.path.exists(os.path.join(out_dir,png_fn)):
       print("%s already exist!" % os.path.join(out_dir,png_fn))
    else:
       boxes_re = []
       masks_re = []
       labels_re = []
       scores_re = []
       cls_id = 0
       class_ids = []
       for i in range(len(imagefiles)):
           #print(mask)
           x1 = float(boxs[i].split('-')[0])
           y1 = float(boxs[i].split('-')[1])
           x2 = float(boxs[i].split('-')[2])
           y2 = float(boxs[i].split('-')[3])
           image = cv2.imread(os.path.join(image_dir, imagefiles[i]))
           (height, width) = image.shape[:2]
           segm = {
                   "size": [height, width], 
                   "counts": masks[i]}
           mask = cocomask.decode(segm)
           if imagefiles[i]==fn:
              boxes_re.extend([(int(y1), int(x1), int(y2), int(x2))])
              masks_re.append(mask)
              labels_re.append(labels[i])
              scores_re.append(scores[i])
              class_ids.append(cls_id)
              cls_id = cls_id + 1
                        
       """
       boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
       masks: [height, width, num_instances]
       class_ids: [num_instances]
       class_names: list of class names of the dataset
       scores: (optional) confidence scores for each box
       title: (optional) Figure title
       show_mask, show_bbox: To show masks and bounding boxes or not
       figsize: (optional) the size of the image
       colors: (optional) An array or colors to use with each object
       captions: (optional) A list of strings to use as captions for each object
       """
       #print(np.array(masks_re).shape)
       image_data = skimage.io.imread(os.path.join(image_dir, fn))
       boxes_plt = np.array(boxes_re)
       masks_re = np.array(masks_re)
       #print(masks_re.shape)
       if len(masks_re.shape) !=1:
           print(image_data.shape)
           print(masks_re.shape[1], masks_re.shape[2], masks_re.shape[0])
            
           masks_plt = np.zeros([masks_re.shape[1], masks_re.shape[2], masks_re.shape[0]])
           for mm in range(masks_re.shape[0]):
               masks_plt[:,:,mm] = masks_re[mm,:,:]
           #masks = masks.reshape(masks.shape[1], masks.shape[2], masks.shape[0])
           class_ids = np.array(class_ids)
           class_names = np.array(labels_re)
           scores_plt = np.array(scores_re)
           #print(boxes.shape[0]) 
           #print(masks.shape[-1])
           #print(class_ids.shape[0])
           #print(class_names)
           #print(image_name) 
           visualize.display_instances(
              image_data, boxes_plt, masks_plt, class_ids,
              class_names, scores_plt,
              show_bbox=True, show_mask=True,
              title="Predictions")
           png_fn = fn.replace('.png', '_pred.png')
           plt.savefig(os.path.join(out_dir,png_fn))
         #plt.

