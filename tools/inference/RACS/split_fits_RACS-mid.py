import os
import os.path as osp

from astropy.io import fits
import numpy as np
import argparse

#getfits -sv [-i num] [-o name] [-d dir] file.fits [x y dx [dy] ...
#x y: Center pixel of region to extract

wcstools_path = '/home/data0/lbq/software/wcstools-3.9.7/bin' 
subimg_exec = '%s/getfits' % wcstools_path 
splitimg_cmd = '{0} -o %s -d %s %s %d %d %d %d'.format(subimg_exec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split_fits')
    parser.add_argument('--fname', dest='fname', type=str, default='/o9000/MWA/GLEAM/data_challenge1/SKAMid_B1_1000h_v3.fits', help='fits file name')
    parser.add_argument('--workdir', dest='workdir', type=str, default='/o9000/MWA/GLEAM/data_challenge1/split_B1_1000h', help='output png dir')
    parser.add_argument('--ImageSize', dest='ImageSize', type=int, default=132, help='size of split image')
    parser.add_argument('--OverlappingSize', dest='OverlappingSize', type=int, default=20, help='overlapping size of split image')
    args = parser.parse_args() 
    fname = args.fname
    work_dir = args.workdir
    ImageSize = args.ImageSize
    OverlappingSize = args.OverlappingSize
    hdu_list = fits.open(fname)
    h1,image_data = hdu_list[0].header, hdu_list[0].data
    print(image_data.shape)
    for i, x1 in enumerate(range(0, image_data.shape[3], ImageSize - OverlappingSize)):
        for j, y1 in enumerate(range(0, image_data.shape[2], ImageSize - OverlappingSize)):
            #print(osp.basename(fname))#(os.path.splitext(fname)[0])
            fid = osp.basename(fname)[21:28] + '_%d-%d.fits' % (i, j)  #osp.basename(fname).replace('.fits', '_%d-%d.fits' % (i, j))
            #out_fname = osp.join(work_dir, fid)
            print(splitimg_cmd % (fid, work_dir, fname, x1, y1, ImageSize, ImageSize))
            os.system(splitimg_cmd % (fid, work_dir, fname, x1, y1, ImageSize, ImageSize))
                    

 
