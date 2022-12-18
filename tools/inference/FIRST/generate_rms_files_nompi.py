import os,sys
import argparse
import linecache
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--inpdir', dest='inpdir', type=str, default='/o9000/MWA/GLEAM/hetu_images/deep_learn/inference_sets/FIRST_fits', help='pred input png file directory')
parser.add_argument('--outdir', dest='outdir', type=str, default='/o9000/MWA/GLEAM/hetu_images/deep_learn/inference_sets/FIRST_rms', help='output rms file directory')
parser.add_argument('--fitslists', dest='fitslists', type=str, help='input fits lists')

args = parser.parse_args()

input_dir = args.inpdir
#file_nms = os.listdir(input_dir)

with open(args.fitslists, 'r') as file:
     linecount=len(file.readlines())

for n in range(linecount):
#for fn in file_nms:
    line = linecache.getline(args.fitslists, n+1)
    fits_file = line.split('\n')[0]

    os.system('BANE %s/%s --out %s/%s' % (input_dir, fits_file, args.outdir, os.path.splitext(fits_file)[0]))
    print("Successful generate %s" % fits_file)
