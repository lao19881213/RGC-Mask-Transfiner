import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.wcs as wcs
from astropy import coordinates
import astropy.units as u
import pandas as pd
import numpy as np
import matplotlib as mpl
import os

def CoolColormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#000069', '#00188a', '#0d6bff', '#1abaff',
                                                                 '#d9ffff', '#ffffff'], 256)

def find_bbox_flux(x1, y1, x2, y2, fitsfile):
    hdu = fits.open(fitsfile)[0]
    print('Processing %s ...' % fitsfile)
    # Set any NaN areas to zero or the interpolation will fail
    hdu.data[np.isnan(hdu.data)] = 0.0

    # Get vital stats of the fits file
    bmaj = hdu.header["BMAJ"]
    bmin = hdu.header["BMIN"]
    bpa = hdu.header["BPA"]
    xmax = hdu.header["NAXIS1"]
    ymax = hdu.header["NAXIS2"]
    try:
        pix2deg = hdu.header["CD2_2"]
    except KeyError:
        pix2deg = hdu.header["CDELT2"]
    # Montaged images use PC instead of CD
    if pix2deg == 1.0:
        pix2deg = hdu.header["PC2_2"]
    beamvolume = (1.1331 * bmaj * bmin)
    print(x1, y1, x2, y2)
    #x1 = float(bbox.split('-')[0])
    #y1 = float(bbox.split('-')[1])
    #x2 = float(bbox.split('-')[2])
    #y2 = float(bbox.split('-')[3])

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    #box_data = hdu.data[hdu.data.shape[0]-y2:hdu.data.shape[0]-y1,x1:x2]
    #for vlass fits data
    if len(hdu.data.shape)==4:
       img_data = hdu.data[0,0,:,:]
    else:
       img_data = hdu.data
    img_data[np.isnan(img_data)] = 0.0
    box_data = img_data[y1:y2,x1:x2]
    if box_data.shape[0] == 0 or box_data.shape[1] ==0:
       print('zero of bbox size: %s' % fitsfile)
       int_flux = 0.0
    else:
       int_flux = np.sum(box_data) #Jy/pix
       int_flux = int_flux * (pix2deg**2) / beamvolume #Jy
       #test
       #w = wcs.WCS(hdu.header, naxis=2)
       #fig = plt.figure()
       #ax = plt.subplot(projection=w)
       #ax.set_xlim([0, box_data.shape[1]])
       #ax.set_ylim([box_data.shape[0], 0])
       #ax.set_axis_off()
       #datamax = np.nanmax(box_data)
       #datamin = np.nanmin(box_data)
       #datawide = datamax - datamin
       #image_data = np.log10((box_data - datamin) / datawide * 1000 + 1) / 3 
       #ax.imshow(image_data, origin='lower', cmap=CoolColormap())
       #pngf = 'J' + fitsfile.split('J')[1].replace('fits', 'png') 
       #plt.savefig(os.path.join('/home/data0/lbq/inference_data/VLASS_flux_test/', pngf))
       ##print(os.path.join('/home/data0/lbq/inference_data/VLASS_flux_test/', pngf))
       #plt.clf()
    return int_flux


hetu_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed.csv')

ras = hetu_csv['centre_ra'].values
decs = hetu_csv['centre_dec'].values
boxs = hetu_csv['box'].values
labels = hetu_csv['label'].values
source_names = hetu_csv['source_name'].values

root_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT'
out_dir = '/home/data0/lbq/inference_data/VLASS_bbox/HT'
VLASS_dir = '%s/VLASS_final' % root_dir
vlass_fitsns = os.listdir(VLASS_dir)

VLASS_bkg_dir = '/home/data0/lbq/inference_data/HT/VLASS_bkg'

final_flux_final = []

for m in range(len(ras)):
      FIRST_fits = '%s/FIRST/%s.fits' % (root_dir, source_names[m])
      hdu_FIRST = fits.open(FIRST_fits)[0] 
      try:
          pix_scale_hetu = hdu_FIRST.header["CD2_2"]
      except KeyError:
          pix_scale_hetu = hdu_FIRST.header["CDELT2"]
     
      items = source_names[m]
      #print(items)
      if(len(items.split('-'))>=2):
        RAlist=items.split('-')[0].split('J')[1]
        Declist=items.split('-')[1]
        ra_first = (float(RAlist[0:2])*15 + float(RAlist[2:4])/60.0*15.0 + float(RAlist[4:])/3600.0*15.0)
        dec_first = 0 - (float(Declist[0:2]) + float(Declist[2:4])/60.0 + float(Declist[4:])/3600.0)
      elif(len(items.split('+'))>=2):
        RAlist=items.split('+')[0].split('J')[1]
        Declist=items.split('+')[1]
        ra_first = (float(RAlist[0:2])*15 + float(RAlist[2:4])/60.0*15.0 + float(RAlist[4:])/3600.0*15.0)
        dec_first = (float(Declist[0:2]) + float(Declist[2:4])/60.0 + float(Declist[4:])/3600.0)
      #print(ra_first, dec_first)
      #sky_first = SkyCoord.from_name(items)
      #ra_first = sky_first.ra.value
      #dec_first = sky_first.dec.value
      #print(ra_first, dec_first)
      for n in range(len(vlass_fitsns)):
          fn = 'J' + vlass_fitsns[n].split('J')[1]
          pngn = fn[0:19]
          # J072959.65+282815.6
          if(len(pngn.split('-'))>=2):
            RAlist=pngn.split('-')[0].split('J')[1]
            Declist=pngn.split('-')[1]
            ra_vlass = (float(RAlist[0:2])*15 + float(RAlist[2:4])/60.0*15.0 + float(RAlist[4:])/3600.0*15.0)
            dec_vlass = 0 - (float(Declist[0:2]) + float(Declist[2:4])/60.0 + float(Declist[4:])/3600.0)
          elif(len(pngn.split('+'))>=2):
            RAlist=pngn.split('+')[0].split('J')[1]
            Declist=pngn.split('+')[1]
            ra_vlass = (float(RAlist[0:2])*15 + float(RAlist[2:4])/60.0*15.0 + float(RAlist[4:])/3600.0*15.0)
            dec_vlass = (float(Declist[0:2]) + float(Declist[2:4])/60.0 + float(Declist[4:])/3600.0) 
          #sky_vlass = SkyCoord.from_name(pngn)
          #ra_vlass = sky_vlass.ra.value
          #dec_vlass = sky_vlass.dec.value
           
          if((abs((ra_first-ra_vlass)*3600.0) <=1.5) and (abs((dec_first-dec_vlass)*3600.0) <=1.5)):
             #print(pngn)
             #print((ra_first-ra_vlass)*3600.0, dec_first - dec_vlass)
             #print('Cross matched FIRST and VLASS: %s, %s' % (items, pngn))
             vlass_fits_f = vlass_fitsns[n]
             break
          else:
             if n == len(vlass_fitsns) - 1:
                 vlass_fits_f = ''
      if vlass_fits_f != '': 
         hdu_vlass= fits.open('%s/%s' % (VLASS_dir, vlass_fits_f))[0]
         #data_vlass = hdu_vlass.data[0,0,:,:]
         #print(hdu_vlass.data.shape)
         #img_vlass = hdu_vlass.data[0,0,:,:]
         w1_vlass=wcs.WCS(hdu_vlass.header, naxis=2)
 
         try:
             pix_scale_vlass = hdu_vlass.header["CD2_2"]
         except KeyError:
             pix_scale_vlass = hdu_vlass.header["CDELT2"]

         x1 = float(boxs[m].split('-')[0])
         y1 = float(boxs[m].split('-')[1])
         x2 = float(boxs[m].split('-')[2])
         y2 = float(boxs[m].split('-')[3])
         width = x2 - x1 + 2
         height = y2 - y1 + 2
         #factor = pix_scale_vlass / pix_scale_hetu
         centre_x, centre_y = w1_vlass.wcs_world2pix([[ras[m],decs[m]]],0).transpose() 
         width_new = width * pix_scale_hetu / pix_scale_vlass
         height_new = height * pix_scale_hetu / pix_scale_vlass
         x1_new = centre_x - width_new/2.0
         x2_new = centre_x + width_new/2.0
         y1_new = centre_y - height_new/2.0
         y2_new = centre_y + height_new/2.0 
         #print(x1_new, x2_new, y1_new, y2_new, img_vlass.shape)
         #plot bbox 
         #if len(hdu_vlass.data.shape)==4:
         #   img_data = hdu_vlass.data[0,0,:,:]
         #else :
         #   img_data = hdu_vlass.data
         #fig = plt.figure()
         #ax = plt.subplot(projection=w1_vlass)
         #ax.set_xlim([0, img_data.shape[1]])
         #ax.set_ylim([img_data.shape[0], 0])
         #ax.set_axis_off()
         #datamax = np.nanmax(img_data)
         #datamin = np.nanmin(img_data)
         #datawide = datamax - datamin
         #image_data = np.log10((img_data - datamin) / datawide * 1000 + 1) / 3

         ##img_rms = 0.00002
         #ax.imshow(image_data, origin='lower', cmap=CoolColormap()) 
         ##boxs_new.append('%s,%.5f-%.5f-%.5f-%.5f' % (source_names[m], x1_new, y1_new, x2_new, y2_new))
         #top, left, bottom, right = y1_new[0], x1_new[0], y2_new[0], x2_new[0]
         ##top, left, bottom, right = img_data.shape[0]-y2_new[0], x1_new[0], img_data.data.shape[0]-y1_new[0], x2_new[0]
         #ax.add_patch(
         #plt.Rectangle((left, top),abs(left - right),abs(top - bottom), \
         #             fill=False, edgecolor='r', linewidth=2)
         #            )
         #plt.savefig(os.path.join(out_dir, '%s.png' % source_names[m]))
         #plt.clf()
         
         #calculate flux density
         #bbox = '{:.5f}-{:.5f}-{:.5f}-{:.5f}'.format(float(x1_new[0]), float(y1_new[0]), float(x2_new[0]), float(y2_new[0]))
         total_flux = find_bbox_flux(float(x1_new[0]), float(y1_new[0]), float(x2_new[0]), float(y2_new[0]), '%s/%s' % (VLASS_dir, vlass_fits_f))
         if total_flux == 0.0:
            final_flux = 0.0
         else:
            bkg_flux = find_bbox_flux(float(x1_new[0]), float(y1_new[0]), float(x2_new[0]), float(y2_new[0]), '%s/%s_bkg.fits' % (VLASS_bkg_dir, os.path.splitext(vlass_fits_f)[0]))
            final_flux = total_flux - bkg_flux 
      else:
         print('Not Found %s' % source_names[m])
         final_flux = 0.0

      final_flux_final.append(final_flux)


hetu_csv['vlass_flux'] = final_flux_final
hetu_csv.to_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass.csv', index = False)
