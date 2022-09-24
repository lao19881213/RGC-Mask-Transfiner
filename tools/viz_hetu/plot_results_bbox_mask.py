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

parser = argparse.ArgumentParser(description='plot v3 mask results')

parser.add_argument('--resultsfile', dest='results_file', type=str, default='/home/blao/RGC-Mask-Transfiner/output_101_3x_deform/inference/coco_instances_results_val.json', help='object detect results json file')
parser.add_argument('--annsfile', dest='anns_file', type=str, default='/home/blao/RGC-Mask-Transfiner/datasets/coco/annotations/instances_val2022.json', help='annotations test json file')


args = parser.parse_args()

results = args.results_file
img_file = args.anns_file

cls = ['cs','fr1','fr2','ht','cj']


with open(img_file, 'r') as fin:
     data_in = json.load(fin)
     for ii in range(len(data_in['images'])):
         boxes = []
         masks = []
         class_ids = []
         class_names = []
         scores = []
         cls_id = 0
         image_name = None
         with open(results, 'r') as fre:
              data_re = json.load(fre)
              for i in range(len(data_re)):
                  cl_id = int(data_re[i]['category_id'])
                  img_id = int(data_re[i]['image_id'])
                  score = float(data_re[i]['score'])
                  box = data_re[i]['bbox']
                  segm = data_re[i]['segmentation']
                  mask = cocomask.decode(segm)
                  x1 = float(box[0])
                  y1 = float(box[1])
                  x2 = x1 + float(box[2])
                  y2 = y1 + float(box[3])
                  #print(mask)
                  if int(data_in['images'][ii]['id'])==img_id:
                     image_name = data_in['images'][ii]['file_name']
                     boxes.extend([(int(y1), int(x1), int(y2), int(x2))])
                     masks.append(mask)
                     class_ids.append(cls_id)
                     cls_id = cls_id + 1
                     class_names.append(cls[cl_id-1])
                     scores.append(score)
                     print(box)
                     
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
         print(np.array(masks).shape)
         image_dir = "/home/blao/RGC-Mask-Transfiner/datasets/coco/val2022"
         image_data = skimage.io.imread(os.path.join(image_dir, image_name))
         boxes = np.array(boxes)
         masks = np.array(masks)
         masks_re = np.zeros([masks.shape[1], masks.shape[2], masks.shape[0]])
         for mm in range(masks.shape[0]):
             masks_re[:,:,mm] = masks[mm,:,:]
         #masks = masks.reshape(masks.shape[1], masks.shape[2], masks.shape[0])
         class_ids = np.array(class_ids)
         class_names = np.array(class_names)
         scores = np.array(scores)
         print(boxes.shape[0]) 
         print(masks.shape[-1])
         print(class_ids.shape[0])
         print(class_names)
         print(image_name) 
         visualize.display_instances(
            image_data, boxes, masks_re, class_ids,
            class_names, scores,
            show_bbox=True, show_mask=True,
            title="Predictions")
         out_dir = '/home/blao/RGC-Mask-Transfiner/vis_hetu_r101_val'
         png_fn = image_name.replace('.png', '_pred.png')
         plt.savefig(os.path.join(out_dir,png_fn))
      #plt.

