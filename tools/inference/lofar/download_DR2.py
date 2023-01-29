import os

with open('links.txt') as f:
     for line in f:
         fn = line.split('/')[6] + "_mosaic-blanked.fits"
         links = line.split('\n')[0]
         os.system("wget %s -O %s --no-check-certificate" % (links, fn))
