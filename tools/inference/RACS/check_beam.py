from astropy.io import fits
import argparse
import pandas as pd
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--partid', dest='partid', type=int, default='1', help='part id')
args = parser.parse_args()

data_dir = '/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid'
input_dir = '%s/part%d' % (data_dir, args.partid)
file_nms = os.listdir(input_dir)
SBID_final = []
bmaj_final = []
bmin_final = []
bpa_final = []
for fn in file_nms:
    if not fn.endswith('.fits'):
       continue

    fits_file = fn
    SBID_final.append(fn[21:28])
    hdu = fits.open(os.path.join(input_dir, fits_file))[0]
    hdr = hdu.header
    bmaj_final.append(hdr['BMAJ'])
    bmin_final.append(hdr['BMIN'])
    bpa_final.append(hdr['BPA'])
beaminfo = pd.DataFrame({'SBID':SBID_final,'BMAJ':bmaj_final,'BMIN':bmin_final,'BPA':bpa_final})

beaminfo.to_csv("%s/part%d_beam.csv" % (data_dir, args.partid),index=False,sep=',')    
