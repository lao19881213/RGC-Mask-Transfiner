# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:08:02 2023

@author: Baoqiang Lao
"""
# Required to see plots when running on mybinder
import matplotlib 
# matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline 

# Python standard-libraries to download data from the web
from urllib.parse import urlencode
from urllib.request import urlretrieve

#Some astropy submodules that you know already
from astropy import units as u
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord
from astropy.io import fits


#only here to display images
from IPython.display import Image

# These are the new modules for this notebook
from astroquery.simbad import Simbad
from astroquery.sdss import SDSS

from astroquery.sdss import SDSS
from astropy import coordinates as coords
from skimage.measure import find_contours
import cv2
from matplotlib.patches import Polygon
import pycocotools.mask as cocomask
from scipy import spatial
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling import models,fitting
import pandas as pd

FIRST_result = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_sdss_ned_flux_fixed_vlass.csv'

hetu_csv = pd.read_csv(FIRST_result)

sdss_names = hetu_csv['SDSS16'].values
source_names = hetu_csv['source_name'].values
sdss_ras = hetu_csv['sdss_ra'].values
sdss_decs = hetu_csv['sdss_dec'].values
sdss_objid = hetu_csv['objID'].values

z_final = []
photo_z_final = []
flux_oiii_final = []
flux_h_alpha_final = []
nii_6584_flux_final = []
sii_6717_flux_final = []
sii_6731_flux_final = []
h_beta_flux_final = []
eqw_oiii_final = []
lum_oiii_final = []
vdisp_final = [] 
M_BH_final = []
CI_final = []
modelmag_r_final = []
modelmag_u_final = [] 
extinction_r_final = []
kcorr_r_final = []
photo_kcorr_r_final = []
absMag_r_final = []
photo_absMag_r_final = []
dn4000_final = []
dn4000_err_final = []
for m in range(len(source_names)): 
    if not pd.isnull(sdss_names[m]):   
       #SDSS J134002.12+003430.9
       print(sdss_names[m])
       sdss_name = sdss_names[m].split(' ')[1] 
       objn_coord = '%sh%sm%ss %sd%sm%ss' % (sdss_name[1:3], sdss_name[3:5], sdss_name[5:10], sdss_name[10:13], sdss_name[13:15], sdss_name[15:19])
       #objn_coord = '%sd %sd' % (sdss_ras[m], sdss_decs[m])
       pos = coords.SkyCoord(objn_coord, frame='icrs')
       #Photometric Redshifts
       p_xid = SDSS.query_region(pos, radius='5 arcsec', data_release=16)

       query = "SELECT\
               TOP 1 p.z, p.kcorrR, p.absMagR \
               FROM Photoz AS p JOIN PhotoObjAll s ON s.objid = p.objid \
               WHERE s.objid = %s" % (p_xid['objid'][0]) #(sdss_objid[m])
       photoz_t = SDSS.query_sql(query, data_release=16)
       if photoz_t != None:
          photo_z_final.append(photoz_t['z'][0])
          photo_kcorr_r_final.append(photoz_t['kcorrR'][0])
          photo_absMag_r_final.append(photoz_t['absMagR'][0])          
       else:
          photo_z_final.append('--')
          photo_kcorr_r_final.append('--')
          photo_absMag_r_final.append('--')
       xid = SDSS.query_region(pos, radius='5 arcsec', spectro=True, data_release=16)
       print(objn_coord)
       if xid != None:
          sp = SDSS.get_spectra(matches=xid, data_release=16)
          #im = SDSS.get_images(matches=xid, band='r')
          #im[0].writeto('J001247.57+004715.8_spec.fits', overwrite=True)
          "The spectrum is stored as a table in the second item of the list."
          "That means that we can get the Table doing the following"
          spectra_data = sp[0][1].data
          
          flux = spectra_data['flux']
          
          #plt.figure(1)
          #plt.plot(10**spectra_data['loglam'], spectra_data['flux'])
          #plt.xlabel('wavelenght (Angstrom)')
          #plt.ylabel('flux (nanomaggies)')
          #plt.title('SDSS spectra of xxx')
          #
          #plt.show()
          
          # The fourth record stores the positions of some emission lines
          
          lines = sp[0][3].data
          
          for n in ['[O_II] 3727', '[O_III] 5007', 'H_alpha']:
              print(n, " ->", lines['LINEWAVE'][lines['LINENAME']==n])
          
          #plt.figure(2)    
          #plt.plot(10**spectra_data['loglam'], spectra_data['flux'], color='black')
          #plt.axvline(x=lines['LINEWAVE'][lines['LINENAME']=='[O_II] 3727'], label=r'O[II]', color='blue')
          #plt.axvline(x=lines['LINEWAVE'][lines['LINENAME']=='[O_III] 5007'], label=r'O[III]', color='red')
          #plt.axvline(x=lines['LINEWAVE'][lines['LINENAME']=='H_alpha'], label=r'H$\alpha$', color='green')
          
          #plt.xlabel('wavelenght (Angstrom)')
          #plt.ylabel('flux (nanomaggies)')
          #plt.title('SDSS spectra of xxx')
          #plt.legend()  
          
          #Calculate F[o_III]  
          # from astroquery.sdss import SDSS
          ##Method 1
          #https://github.com/search?q=sdss+oiii_5007+luminosity&type=code
          for mm in range(len(xid['specobjid'])):
              query = "SELECT\
                      TOP 1 g.oiii_5007_flux, g.oiii_5007_flux_err,  \
                      oiii_5007_eqw, oiii_5007_eqw_err, \
                      h_alpha_flux, h_alpha_flux_err \
                      FROM GalSpecLine AS g  JOIN SpecObj AS s  ON s.specobjid = g.specobjid \
                      WHERE s.specobjid = %s" % (xid['specobjid'][mm])
              #query = "SELECT\
              #        TOP 1 l.oiii_5007_flux, l.oiii_5007_flux_err \
              #        FROM SpecObjAll AS s \
              #        JOIN GalSpecInfo AS g ON s.specobjid = g.specobjid \
              #        JOIN GalSpecLine AS l ON s.specobjid = l.specobjid \
              #        JOIN PhotoObjAll as p on s.bestobjid = p.objid \
              #        WHERE p.objid = %s" % (xid['objid'][mm])
                          #WHERE spec.ra = 2.02629 AND spec.dec = 0.011883"
              
              # query = "SELECT TOP 1 s.plate, s.mjd, s.fiberid, s.z, g.subclass, g.e_bv_sfd, l.oiii_5007_flux \
              #     FROM SpecObjAll AS s JOIN GalSpecInfo AS g ON s.specobjid = g.specobjid \
              #     JOIN GalSpecLine AS l ON s.specobjid = l.specobjid WHERE g.specobjid = %s" % (xid['specobjid'][0])
              results = SDSS.query_sql(query, data_release=16)
              if results != None:
                 #in erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$) / 1e-17
                 oiii_5007_flux = results['oiii_5007_flux'][0]
                 h_alpha_flux = results['h_alpha_flux'][0]
                 oiii_5007_eqw = results['oiii_5007_eqw'][0]
                 if oiii_5007_flux <= 0.0:
                    oiii_5007_flux = 1.4826 * np.median(abs(flux - np.median(flux)))
                 print('flux of oIII from database -> ', oiii_5007_flux)
                 print('flux of h_alpha from database -> ', results['h_alpha_flux'][0])
                 print('equivalent width of oIII from database -> ', results['oiii_5007_eqw'][0])
                 results_y_1 = 1 
                 break
              else:
                 results_y_1 =0
          if results_y_1 == 0:
              oiii_5007_flux = '--'
              h_alpha_flux = '--'
              oiii_5007_eqw = '--'
          
          flux_oiii_final.append(oiii_5007_flux)     
          flux_h_alpha_final.append(h_alpha_flux)
          eqw_oiii_final.append(oiii_5007_eqw)

          for mm in range(len(xid['specobjid'])):
              query = "SELECT\
                      TOP 1 nii_6584_flux, nii_6584_flux_err, \
                      sii_6717_flux, sii_6717_flux_err, \
                      sii_6731_flux, sii_6731_flux_err, \
                      h_beta_flux, h_beta_flux_err \
                      FROM GalSpecLine AS g  JOIN SpecObj AS s  ON s.specobjid = g.specobjid \
                      WHERE s.specobjid = %s" % (xid['specobjid'][mm])
              results_1 = SDSS.query_sql(query, data_release=16)
              if results_1 != None:
                 nii_6584_flux_final.append(results_1['nii_6584_flux'][0])
                 sii_6717_flux_final.append(results_1['sii_6717_flux'][0])
                 sii_6731_flux_final.append(results_1['sii_6731_flux'][0])
                 h_beta_flux_final.append(results_1['h_beta_flux'][0])
                 print('flux of nii_6584_flux from database -> ', results_1['nii_6584_flux'][0])
                 print('flux of sii_6717_flux from database -> ', results_1['sii_6717_flux'][0])
                 print('flux of sii_6731_flux from database -> ', results_1['sii_6731_flux'][0])
                 print('flux of h_beta_flux from database -> ', results_1['h_beta_flux'][0])
                 results_y_2 = 1 
                 break
              else:
                 results_y_2 =0
          if results_y_2 == 0:
              nii_6584_flux_final.append('--')
              sii_6717_flux_final.append('--')
              sii_6731_flux_final.append('--')
              h_beta_flux_final.append('--')
          #Method 2 
          float_lam = 10**spectra_data['loglam'] 
          int_lam =  float_lam.astype(np.int64)
          x=lines['LINEWAVE'][lines['LINENAME']=='[O_III] 5007']
          index_oIII = np.where(int_lam==int(x))[0]
          flux_oIII = flux[index_oIII][0]
          print('flux of oIII -> ',flux_oIII)
          
          
          #plt.figure(3)
          #hb_sliced_index = np.where(np.logical_and(float_lam>=4950, float_lam<=5050))
          #w_hb = float_lam[hb_sliced_index]
          #flux_hb = flux[hb_sliced_index]
          #plt.plot(w_hb, flux_hb, color='black')
          
          #plt.figure(4)
          #
          ## use AstroPy to define our models, and give initial guesses for the Gaussian parameters
          #cont = models.Polynomial1D(1)
          #g1 = models.Gaussian1D(amplitude=110, mean=4990, stddev=10)
          #
          ## define the total model to fit to our data: the continuum and emission line
          #g_total = g1 + cont
          #
          ## define the fitter
          #fit_g = fitting.LevMarLSQFitter()
          ## fit the model to the data
          #g = fit_g(g_total, w_hb, flux_hb, maxiter = 1000)
          ##print(g)
          #
          ## define an array of wavelength values across our wavelength range with
          ##a high number of steps, to plot our model over our data
          #x_g = np.linspace(np.min(w_hb), np.max(w_hb), 10000)
          #
          ## plot the model and the data
          #plt.figure(figsize=(7,7))
          #plt.step(w_hb, flux_hb, where='mid', color='k') # plot the data
          #plt.plot(x_g, g(x_g), color='red')
          #plt.xlabel(r"Wavelength ($\AA$)", fontsize=18)
          #plt.ylabel(r"Flux (erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$) / 1e-17", fontsize=18)
          #plt.xticks(fontsize=16)
          #plt.yticks(fontsize=16)
          #
          #plt.show()
          
          #redshift
          redshift = sp[0][2].data['Z'][0]
          print('redshift -> ', redshift)
          z_final.append(redshift)
           
          # RA and DEC
          #ra=sp[0][0].header['RA'] ; dec=sp[0][0].header['DEC']
          
          #velocity dispersion in km/s
          vdisp = sp[0][2].data['VDISP'][0]
          print('Stellar velocity dispersion -> ', vdisp)
          vdisp_final.append(vdisp) 
          # black hole of mass
          if vdisp >0:
             M_BH = 8.13 + 4.02*np.log10(vdisp/200.0)
          else:
             M_BH = 8.13
          print('black hole of mass -> ', M_BH)
          M_BH_final.append(M_BH) 
          #Concentration index
          #xid['specobjid']
          for mm in range(len(xid['specobjid'])): 
              query = "SELECT\
                      TOP 1 p.petroR50_r, p.petroR90_r \
                      FROM PhotoObjAll AS p JOIN specObjAll s ON s.bestobjid = p.objid \
                      WHERE s.specobjid = %s" % (xid['specobjid'][0]) 
              SDSSpetro = SDSS.query_sql(query, data_release=16)
              if SDSSpetro != None:
                 Ci = SDSSpetro['petroR90_r'][mm]/SDSSpetro['petroR50_r'][mm]
                 print('Concentration index -> ', Ci)
                 SDSSpetro_y = 1
                 break
              else:
                 SDSSpetro_y = 0
          if SDSSpetro_y == 0:
             Ci = '--'
          CI_final.append(Ci)       
          #the [OIII] line luminosity 
          if results != None:
             if results['oiii_5007_flux'][0] > 0:
                #Planck 2018 results. VI. Cosmological parameters 
                cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)
                z=redshift
                distance = cosmo.luminosity_distance(z).value * 3.08567758*1e+24
                #in erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$
                flux_oiii = results['oiii_5007_flux'][0] *10**(-17)
                # wave = 5008.23963773
                # flux_oiii = flux_oiii * 3e+18 / wave**2. * wave
                lum = np.log10(flux_oiii * 4. * np.pi * distance**2.)
                print('[OIII] line luminosity -> ', lum)
             else:
                flux_rms = 1.4826 * np.median(abs(flux - np.median(flux)))
                lum = np.log10(flux_rms * 4. * np.pi * distance**2.)
                print('[OIII] line luminosity -> ', lum)
                #lum = '--'  
          else:
             lum = '--'
          lum_oiii_final.append(lum)
          #Mr
          for mm in range(len(xid['specobjid'])):          
              query= "SELECT\
                     TOP 1 p.objid, p.ra, p.dec, p.modelmag_r, p.modelmag_u, p.extinction_r \
                     FROM SpecPhoto AS p JOIN specObjAll s ON s.bestobjid = p.objid \
                     WHERE s.specobjid = %s" % (xid['specobjid'][mm])
              mq = SDSS.query_sql(query, data_release=16)
              if mq != None:
                 modelmag_r_final.append(mq['modelmag_r'][0])
                 modelmag_u_final.append(mq['modelmag_u'][0]) 
                 extinction_r_final.append(mq['extinction_r'][0])
                 print('modelmag_r -> ', mq['modelmag_r'][0])
                 print('modelmag_u -> ', mq['modelmag_u'][0])
                 print('extinction_r -> ', mq['extinction_r'][0])
                 mq_y = 1
                 break
              else:
                 mq_y = 0
          if mq_y == 0:
              modelmag_r_final.append('--')
              modelmag_u_final.append('--')
              extinction_r_final.append('--') 
          for mm in range(len(xid['specobjid'])):
              query = "SELECT\
                       TOP 1 p.kcorrR, p.absMagR, p.z \
                       FROM Photoz AS p JOIN specObjAll s ON s.bestobjid = p.objid \
                       WHERE s.specobjid = %s" % (xid['specobjid'][mm]) 
              mq_2 = SDSS.query_sql(query, data_release=16)
              if mq_2 != None:
                 absMag_r_final.append(mq_2['absMagR'][0])
                 kcorr_r_final.append(mq_2['kcorrR'][0])
                 print('absMag_r -> ', mq_2['absMagR'][0])
                 print('kcorr_r -> ', mq_2['kcorrR'][0])
                 mq_2_y = 1
                 break
              else:
                 mq_2_y = 0
          if mq_2_y == 0:
             absMag_r_final.append('--')
             kcorr_r_final.append('--')
          #Dn(4000)
          for mm in range(len(xid['specobjid'])):
              query = "SELECT\
                       TOP 1 g.d4000_n, g.d4000_n_err \
                       FROM galSpecIndx AS g JOIN SpecObj AS s  ON s.specobjid = g.specobjid \
                       WHERE s.specobjid = %s" % (xid['specobjid'][mm])
              dn = SDSS.query_sql(query, data_release=16)
              if dn != None:
                 dn4000_final.append(dn['d4000_n'][0])
                 dn4000_err_final.append(dn['d4000_n_err'][0])
                 print('dn4000 -> ', dn['d4000_n'][0])
                 dn_y = 1
                 break
              else:
                 dn_y = 0
          if dn_y == 0:
             dn4000_final.append('--')
             dn4000_err_final.append('--')
          
       else:
          z_final.append('--')
          #photo_z_final.append('--')
          flux_oiii_final.append('--')
          lum_oiii_final.append('--')
          vdisp_final.append('--')
          M_BH_final.append('--')
          CI_final.append('--')
          flux_h_alpha_final.append('--')
          eqw_oiii_final.append('--')
          modelmag_r_final.append('--')
          modelmag_u_final.append('--')
          extinction_r_final.append('--')
          absMag_r_final.append('--')
          kcorr_r_final.append('--')
          dn4000_final.append('--')
          dn4000_err_final.append('--')
          nii_6584_flux_final.append('--')
          sii_6717_flux_final.append('--')
          sii_6731_flux_final.append('--')
          h_beta_flux_final.append('--')
    else:
       z_final.append('--')
       photo_z_final.append('--')
       flux_oiii_final.append('--')
       lum_oiii_final.append('--')
       vdisp_final.append('--')
       M_BH_final.append('--')
       CI_final.append('--')
       flux_h_alpha_final.append('--')
       eqw_oiii_final.append('--')
       modelmag_r_final.append('--')
       modelmag_u_final.append('--')
       extinction_r_final.append('--')
       absMag_r_final.append('--')
       kcorr_r_final.append('--')
       dn4000_final.append('--')
       dn4000_err_final.append('--')
       nii_6584_flux_final.append('--')
       sii_6717_flux_final.append('--')
       sii_6731_flux_final.append('--')
       h_beta_flux_final.append('--')
       photo_kcorr_r_final.append('--')
       photo_absMag_r_final.append('--')


hetu_csv['z'] = z_final
hetu_csv['photo_z'] = photo_z_final
hetu_csv['flux_oiii'] = flux_oiii_final
hetu_csv['lum_oiii'] = lum_oiii_final 
hetu_csv['vdisp'] = vdisp_final
hetu_csv['M_BH'] = M_BH_final
hetu_csv['Ci'] = CI_final
hetu_csv['flux_h_alpha'] = flux_h_alpha_final 
hetu_csv['eqw_oiii'] = eqw_oiii_final
hetu_csv['modelmag_r'] = modelmag_r_final
hetu_csv['modelmag_u'] = modelmag_u_final
hetu_csv['extinction_r'] = extinction_r_final
hetu_csv['kcorr_r'] = extinction_r_final
hetu_csv['absMag_r'] = absMag_r_final
hetu_csv['photo_kcorr_r'] = photo_kcorr_r_final
hetu_csv['photo_absMag_r'] = photo_absMag_r_final
hetu_csv['dn4000'] = dn4000_final
hetu_csv['dn4000_err'] = dn4000_err_final
hetu_csv['nii_6584_flux'] = nii_6584_flux_final
hetu_csv['sii_6717_flux'] = sii_6717_flux_final
hetu_csv['sii_6731_flux'] = sii_6731_flux_final
hetu_csv['h_beta_flux'] = h_beta_flux_final 
hetu_csv.to_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_sdss_ned_flux_fixed_vlass_optical_spectra.csv', index = False)

