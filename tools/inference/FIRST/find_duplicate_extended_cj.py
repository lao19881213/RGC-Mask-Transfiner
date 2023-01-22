import os
import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
from PIL import Image
import argparse
import pandas as pd
import linecache
import gc

def find():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fr12csv', help='fr12 csv file')
    parser.add_argument('--inpdir', help='input fits and png file directory')
    parser.add_argument('--outdir', help='output results directory')
    parser.add_argument('--outfile', help='output file')
    parser.add_argument('--rank', help='rank')
    args = parser.parse_args()
    #tags = []
    pd.set_option('mode.chained_assignment', None)
    os.system('rm %s/%s' % (args.outdir, args.outfile))
    
    input_dir = args.inpdir
    
    fr12_csv = args.fr12csv
    
    csv_fr12 = pd.read_csv(fr12_csv)

    ras = csv_fr12['centre_ra'].values

    decs = csv_fr12['centre_dec'].values

    boxs = csv_fr12['box'].values

    image_files = csv_fr12['imagefilename'].values

    len_s = len(ras)
    #del csv_fr12
    #gc.collect()
    pro_arr = np.array_split(np.arange(len(image_files)),8)
    print(len(pro_arr[0])+len(pro_arr[1])+len(pro_arr[2])+len(pro_arr[3])+len(pro_arr[4]) \
         +len(pro_arr[5])+len(pro_arr[6])+len(pro_arr[7]))
    for n in pro_arr[int(args.rank)]: 
    #for n in range(len_s):
       print(n)
       rep_row = []
       out_f = args.outfile 
       fits_file = os.path.splitext(image_files[n])[0] + ".fits"
       hdu = fits.open(os.path.join(input_dir, fits_file))
       image = hdu[0].data
       w = wcs.WCS(hdu[0].header, naxis=2)
       width = hdu[0].data.shape[1]
       height = hdu[0].data.shape[0]
    
       bottom_left = [0, 0]
       top_left = [0, height - 1]
       top_right = [width - 1, height - 1]
       bottom_right = [width - 1, 0]
    
       ret = np.zeros([4, 2])
       ret[0, :] = w.wcs_pix2world([bottom_left], 0)[0][0:2]
       ret[1, :] = w.wcs_pix2world([top_left], 0)[0][0:2]
       ret[2, :] = w.wcs_pix2world([top_right], 0)[0][0:2]
       ret[3, :] = w.wcs_pix2world([bottom_right], 0)[0][0:2]
       RA_min, DEC_min, RA_max, DEC_max = np.min(ret[:, 0]),   np.min(ret[:, 1]),  np.max(ret[:, 0]),  np.max(ret[:, 1])
    
       x1 = float(boxs[n].split('-')[0])
       y1 = float(boxs[n].split('-')[1])
       x2 = float(boxs[n].split('-')[2])
       y2 = float(boxs[n].split('-')[3])
    
       x1 = int(x1)
       y1 = int(y1)
       x2 = int(x2)
       y2 = int(y2)
       #print(x1,x2,y1,y2)
       y1_new = hdu[0].data.shape[0]-y2
       y2_new = hdu[0].data.shape[0]-y1
       
        
       for m in range(n+1,len_s):
           ra_first = float(ras[m])
           dec_first = float(decs[m])
           img_x, img_y = w.wcs_world2pix([[ra_first,dec_first]],0).transpose()
           if ra_first <= RA_max and ra_first >= RA_min and dec_first <= DEC_max and dec_first >= DEC_min:
              if img_x <= x2 and img_x >= x1 and img_y <= y2_new and img_y >= y1_new:
                 rep_row.append(m)
           del ra_first
           del dec_first
           del img_x, img_y
           #print(gc.collect()) 
           gc.collect(0)
           gc.collect(1)
           #gc.collect(2)
       str_row = ", ".join(str(i) for i in rep_row)
       cmd_str = '%d, %s, %s, %s, %s, %s' % (n, image_files[n], boxs[n], ras[n], decs[n], str_row)
       print(cmd_str)
       #with open(os.path.join(args.outdir, out_f), 'a') as f:
       #     f.write(cmd_str)
       os.system("echo '%s' >> %s/%s" % (cmd_str, args.outdir, out_f))      
       del x1
       del y1
       del x2
       del y2
       del y1_new
       del y2_new
       del w
       #del ra_first
       #del dec_first
       del rep_row
       #del img_x, img_y
       #del m
       gc.collect()

if __name__ == '__main__':
    find()
#with open(os.path.join(args.outdir, args.outfile), 'w') as f:
#     f.write(os.linesep.join(tags))
