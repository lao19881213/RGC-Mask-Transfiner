import os
import argparse
"""
Convert fits file into miriad file for fitting
"""

cmd_t = 'fits in=%s/%s op=xyin  out=%s/%s'

def convert(fits_dir, fn, mir_dir):
    cmd = cmd_t % (fits_dir, fn, mir_dir, fn.replace('.fits', '.mir'))
    print(cmd)
    os.system(cmd)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fits2mir')
    parser.add_argument('--fitsdir', dest='fitsdir', type=str, default='/home/blao/RGC-Mask-Transfiner/datasets/coco/LOFAR', help='input fits file dir')
    parser.add_argument('--fitslist', dest='fitslist', type=str, default='/home/blao/RGC-Mask-Transfiner/datasets/coco/lofar.txt', help='input fits list')
    parser.add_argument('--mirdir', dest='mirdir', type=str, default='/home/blao/RGC-Mask-Transfiner/datasets/coco/LOFAR_mir', help='output mir dir')
    args = parser.parse_args()

    fits_dir = args.fitsdir
    fits_list = args.fitslist
    with open(fits_list) as f:
         for fn in f.readlines():
             png_file = fn.split('\n')[0]
             fits_file = png_file.replace('.png', '.fits')
             mir_dir = args.mirdir
             convert(fits_dir, fits_file, mir_dir)
