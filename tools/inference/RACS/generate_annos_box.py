import os
import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
from PIL import Image
import matplotlib as mpl
import argparse
#from mpi4py import MPI
import cv2
import json
import base64
import pandas as pd 

parser = argparse.ArgumentParser()
parser.add_argument('--result', help='result file')
parser.add_argument('--inpdir', help='pred input png file directory')
parser.add_argument('--outdir', help='pred output png file directory')
args = parser.parse_args()

result_file = args.result

hetu = pd.read_csv(result_file)

img_fn_hetu = hetu['imagefilename'].values
boxs = hetu['box'].values
labels = hetu['label'].values
scores = hetu['score'].values
 

input_dir = args.inpdir
out_dir = args.outdir 
file_nms = os.listdir(input_dir)
#comm=MPI.COMM_WORLD
#num_process=comm.Get_size()
#rank=comm.Get_rank()

#pro_arr = np.array_split(np.arange(len(file_nms)),num_process)
for n in range(len(file_nms)):#pro_arr[rank]:
    if not file_nms[n].endswith('.png'):
       continue
    score_re = [] 
    predicted_class_re = []
    score_re = []
    box_re = []
   
    #with open(result_file,'r') as f:
    for m in range(len(img_fn_hetu)):
        image_file = img_fn_hetu[m]#line.split(',')[0]
        if image_file == file_nms[n] :
           #print(image_file, file_nms[n])
           box = boxs[m] #line.split(',')[3] 
           box_re.append(box)
           predicted_class = labels[m]#line.split(',')[1] 
           predicted_class_re.append(predicted_class)
           score = float(scores[m])#float(line.split(',')[2])
           score_re.append(score)
    if len(box_re) > 0:
       json_file = 'example.json'
       image = cv2.imread(os.path.join(input_dir, file_nms[n]))

       (height, width) = image.shape[:2]

       with open(json_file,'r')as f:
            json_data = json.load(f)
            #shapes = json_data['shapes']
            with open(os.path.join(input_dir, file_nms[n]), 'rb') as img_f:
                 image_data = img_f.read()
                 image_bytes = base64.b64encode(image_data)
                 image_tring = image_bytes.decode('utf-8')
                 json_data['imageData'] = image_tring
                 json_data['imageHeight'] = height
                 json_data['imageWidth'] = width
                 json_data['imagePath'] = '%s' % file_nms[n] 
                 json_data['shapes'] = []
            for m in range(len(box_re)):#(shapes)):
                 x1 = float(box_re[m].split('-')[0])
                 y1 = float(box_re[m].split('-')[1])
                 x2 = float(box_re[m].split('-')[2])
                 y2 = float(box_re[m].split('-')[3])
                 label = "%s" % predicted_class_re[m]
                 json_data['shapes'].append({
                 "label": label,
                 "points": [[x1, y1],[x2, y2]],
                 "group_id": None,
                 "shape_type": "rectangle",
                 "flags": {}
                 })
       fn_json = file_nms[n].replace('.png', '.json')
       with open(os.path.join(out_dir, fn_json),'w')as dump_f:
            json.dump(json_data, dump_f)

