import os
import sys

data_dir = '../HT/FIRST'
fns = os.listdir(data_dir)

#FIRST__J235959.7+004210.7_post?RA=23+59+59.7++00+42+10.7496&Equinox=J2000&ImageSize=4.0&ImageType=FITS+Image&Download=1&FITS=1_s2.0arcmin_.fits
for fn in fns:
    if fn.endswith('arcmin_.fits'):
       print(fn)
       objn = fn.split('_post')[0].split('__')[1]
       print(objn)
       cmd = 'cp %s/%s %s/FIRST_%s.fits' % (data_dir, fn, data_dir, objn)
       cmd = cmd.replace('&', '\&')
       print(cmd)
       os.system(cmd)
       cmd = 'rm %s/%s' % (data_dir, fn)
       cmd = cmd.replace('&', '\&')
       os.system(cmd)

    

