import csv
import numpy as np
import os
import math
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inpdir', help='input labelme json file directory')
parser.add_argument('--outdir', help='output csv directory')
parser.add_argument('--csvfn', help='output csv directory')

args = parser.parse_args()

input_dir =args.inpdir

file_nms = os.listdir(input_dir)

labels = []
fns = []
boxs = []
for n in range(len(file_nms)):
    if not file_nms[n].endswith('.json'):
       continue
    json_file = file_nms[n]
    with open(os.path.join(input_dir, json_file),'r')as f:
         json_data = json.load(f)
         #json_data['shapes']
         len_label = len(json_data['shapes'])
         for mm in range(len_label):
             label = json_data['shapes'][mm]['label']
             x1 = json_data['shapes'][mm]['points'][0][0]
             y1 = json_data['shapes'][mm]['points'][0][1]
             x2 = json_data['shapes'][mm]['points'][1][0]
             y2 = json_data['shapes'][mm]['points'][1][1]
             box='%f-%f-%f-%f' % (x1,y1,x2,y2)
             fns.append(json_file.replace('json', 'png'))
             labels.append(label) 
             boxs.append(box) 


with open(os.path.join(args.outdir, args.csvfn), 'w') as f:
     csv_w = csv.writer(f)
     #filename, labelname, score, box(xmin-ymin-xmax-ymax), local_rms, peak_flux, err_peak_flux, int_flux, err_int_flux, ra, dec, major, err_major, minor, err_minor, pa, err_pa
     csv_w.writerow(["imagefilename", "label", "box"])
     for m in range(len(labels)):
         csv_w.writerow([fns[m], labels[m], boxs[m]])
