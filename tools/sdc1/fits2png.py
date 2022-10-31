import os, math
import os.path as osp

import numpy as np

import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse


def fits2png(fits_dir, png_dir, myrank, total_proc):
    """
    Convert fits to png files based on the D1 method
    """
    cmd_tpl = '%s -cmap Heat'\
        ' -zoom to fit -scale asinh -scale mode minmax -export %s -exit'
    # cmd_tpl = '%s -cmap gist_heat -cmap value 0.684039 0'\
    #     ' -zoom to fit -scale log -scale mode minmax -export %s -exit'
    from sh import Command
    ds9_path = '/home/app/ds9/8.2/cpu/bin/ds9'#'/home/app/ds9/bin/ds9'
    ds9 = Command(ds9_path)
    file_nms = os.listdir(fits_dir)
    file_nms = file_nms[myrank:][::total_proc]
    for fits in file_nms:
        if (fits.endswith('.fits')):
            png = fits.replace('.fits', '.png')
            if os.path.exists(osp.join(png_dir, png)):
               print("%s was exists.\n"  % osp.join(png_dir, png))
            else:
               cmd = cmd_tpl % (osp.join(fits_dir, fits), osp.join(png_dir, png))
               print(cmd)
               #os.system(cmd)
               ds9(*(cmd.split())) #need to open x11
               #print(ds9(*(cmd.split())))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data_prep')

    parser.add_argument('--rank', dest='rank', type=int, default='1', help='rank')
    parser.add_argument('--totalproc', dest='totalproc', type=int, default='1', help='total proc')
     
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = "/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/sdc1/data" #os.path.join(cur_dir, '..', 'data')
    fits_cutout_dir = "/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/sdc1/data/split_B2_1000h" #osp.join(data_dir, 'split_B5_1000h_train')
    png_dir = osp.join(data_dir, 'split_B2_1000h_png')
    args = parser.parse_args()
    myrank = args.rank - 1
    total_proc = args.totalproc
    fits2png(fits_cutout_dir, png_dir, myrank, total_proc)
