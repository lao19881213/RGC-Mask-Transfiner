# -*- coding: utf-8 -*-
"""
Created on Sat May 13 23:26:23 2023

@author: 86185
"""

import numpy as np
import os
import math
import json
import csv
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                                AutoMinorLocator)
from shapely import geometry

def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)

json_data = []
with open('coco_instances_results.json','r')as f:
     # for line in f.readlines():
     json_data = json.load(f)
     #json_data.append(dic)
         

with open('instances_val2022.json', 'r') as fin:
     json_data_truth = json.load(fin)
     
anns = json_data_truth['annotations'] 

image_num = len(json_data_truth['images'])

class_ids = []
for m in range(len(anns)):
     class_ids.append(json_data_truth['annotations'][m]['category_id'])     

class_ids = np.array(class_ids)

for c_id in range(1,6):
    exec('cid_total{} = len(class_ids[class_ids=={}])'.format(c_id, c_id))

pred_scores = []
pred_labels = []
pred_bboxs = []
pred_imageids = []
truth_labels = []
truth_bboxs = []

def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]

        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, (xx2 - xx1))
        h = np.maximum(0.0, (yy2 - yy1))
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

     
        ids = np.where(iou <= threshold)[0]

        order = order[ids + 1]

    return keep

truth_id = []
# pred_data = [] 
for m in range(len(json_data)):
    
    pred_scores.append(json_data[m]['score'])
    pred_labels.append(json_data[m]['category_id'])
    pred_bboxs.append('%.5f,%.5f,%.5f,%.5f' % (json_data[m]['bbox'][0], 
                       json_data[m]['bbox'][1], 
                       json_data[m]['bbox'][0] + json_data[m]['bbox'][2],
                       json_data[m]['bbox'][1] + json_data[m]['bbox'][3]))
    pred_imageids.append(json_data[m]['image_id'])
#     pred_data.append([json_data[m]['image_id'], 
#                      json_data[m]['category_id'],
#                      [json_data[m]['bbox'][0], 
#                        json_data[m]['bbox'][1], 
#                        json_data[m]['bbox'][0] + json_data[m]['bbox'][2],
#                        json_data[m]['bbox'][1] + json_data[m]['bbox'][3]],
#                      json_data[m]['score']])
    

# pred_data = np.array(pred_data)  

with open('pred_results.csv', 'w', newline='') as fcsv:
     csv_w = csv.writer(fcsv)
     #filename, labelname, score, box(xmin-ymin-xmax-ymax), local_rms, peak_flux, err_peak_flux, int_flux, err_int_flux, ra, dec, major, err_major, minor, err_minor, pa, err_pa
     csv_w.writerow(["image_id", "category_id", "bbox", "score"])
     for m in range(len(pred_imageids)):
         csv_w.writerow([pred_imageids[m], pred_labels[m], pred_bboxs[m], pred_scores[m]])  
    

pred_csv = pd.read_csv('pred_results.csv')

# scores_cs = []
# labels_cs = [] 

for c_id in range(1,6):
    for thre in [0.5, 0.75]:
        if thre==0.5:
            thre_str = '50'
        else:
            thre_str = '75'
        exec('pred_cid{}_thre{} = []'.format(c_id, thre_str))

for c_id in range(1,6):   
    print(c_id)
    for img_id in range(1, image_num+1): #image_num+1 #& pred_csv['category_id']==c_id
        pred_same = pred_csv.loc[(pred_csv['image_id']==img_id) & (pred_csv['category_id']==c_id)]
        if len(pred_same)==0:
            continue
        bboxes_tmp =  np.asarray(pred_same['bbox'].values)
        bboxes = []
        for nn in range(len(bboxes_tmp)):
            bboxes.append([float(bboxes_tmp[0].split(',')[0]),
                           float(bboxes_tmp[0].split(',')[1]),
                           float(bboxes_tmp[0].split(',')[2]),
                           float(bboxes_tmp[0].split(',')[3])])
        bboxes = np.asarray(bboxes)
        scores = np.asarray(pred_same['score'].values)
        
        if(bboxes.shape[0]==1):
            proposals = bboxes[0]
            proposals = proposals.T.tolist()
            proposals_score = scores[0]
            x_center = proposals[0] + (proposals[2] - proposals[0])/2.0
            y_center = proposals[1] + (proposals[3] - proposals[1])/2.0
            pt = (x_center, y_center)
            for n in range(len(anns)):
                if img_id == anns[n]['image_id']:
                    xmin = anns[n]['bbox'][0]
                    xmax = anns[n]['bbox'][0] + anns[n]['bbox'][2]
                    ymin = anns[n]['bbox'][1]
                    ymax = anns[n]['bbox'][1] + anns[n]['bbox'][3]
                    square = [(xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax)]
                    if(if_inPoly(square, pt)):
                        truth_label = anns[n]['category_id']
                        break
            if proposals_score>=0.5:
               thre_str = '50'
               exec('pred_cid{}_thre{}.append({})'.format(c_id, thre_str, [img_id, c_id, proposals, proposals_score, truth_label]))
            if proposals_score>=0.75:
               thre_str = '75'
               exec('pred_cid{}_thre{}.append({})'.format(c_id, thre_str, [img_id, c_id, proposals, proposals_score, truth_label]))
        else:
            keep = nms(bboxes, scores)
            proposals = bboxes[keep][0]
            proposals = proposals.T.tolist()
            proposals_score = scores[keep][0]
            if proposals_score>=0.5:
               thre_str = '50'
               exec('pred_cid{}_thre{}.append({})'.format(c_id, thre_str, [img_id, c_id, proposals, proposals_score, truth_label]))
            if proposals_score>=0.75:
               thre_str = '75'
               exec('pred_cid{}_thre{}.append({})'.format(c_id, thre_str, [img_id, c_id, proposals, proposals_score, truth_label]))

#import torch    
for c_id in range(1,6):
    for thre in [50, 75]:
        exec('TP = 0')
        exec('pred_cid{}_thre{} = np.array(pred_cid{}_thre{})'.format(c_id, thre, c_id, thre))
        exec('labels=pred_cid{}_thre{}[:,4]'.format(c_id,thre))
        exec('prediction=pred_cid{}_thre{}[:,1]'.format(c_id,thre))
        exec('TP = [TP+1 for mm in range(len(prediction)) if labels[mm] == prediction[mm]]')
        exec('TP = sum(TP)')
            # if labels[mm] == prediction[mm]:
            #     TP = TP + 1
        # exec('TP = prediction.eq(labels.view_as(prediction)).numpy().sum()')
        exec('num_prediction=len(pred_cid{}_thre{})'.format(c_id,thre))
        exec('P = TP/num_prediction')
        exec('R = TP/cid_total{}'.format(c_id))
        exec('Precision_cid{}_thre{} = P'.format(c_id, thre))
        exec('Recall_cid{}_thre{} = R'.format(c_id, thre))
        exec('print("Pecision class {}, the {}: %.4f" % (Precision_cid{}_thre{}))'.format(c_id, thre,c_id, thre))
        exec('print("Recall class {}, the {}: %.4f" % (Recall_cid{}_thre{}))'.format(c_id, thre, c_id, thre))