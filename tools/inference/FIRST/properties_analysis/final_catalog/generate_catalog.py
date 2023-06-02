import pandas as pd
import numpy as np
import collections

data_dir = "/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results"

sdss_name_final = []
ned_type_final = []
ned_name_final = []
redshift_final = []
panstarrs_ra_final = []
panstarrs_dec_final = []
optical_final = []

for cln in ['fr1']:#, 'fr2', 'ht', 'cj']:
    ned_table = pd.read_csv('../NED/centre_peaks_5arcsec/info_%s.csv' % cln)
    ned_hetu_name = ned_table['HeTu_name'].values
    #ned_total_flux = ned_table['total_flux'].values
    ned_source_name = ned_table['source_name'].values
    ned_type = ned_table['Type'].values
    ned_Redshift = ned_table['Redshift'].values
    ned_num = ned_table['HeTu_num'].values

    hetu = pd.read_csv('%s/FIRST_HeTu_paper_%s.csv' % (data_dir, cln))
    hetu_source_name = hetu['source_name'].values
    hetu_total_flux = hetu['int_flux'].values
    hetu_box = hetu['box'].values
    sdss = pd.read_csv('%s/FIRST_HeTu_paper_%s_SDSS_DR16.csv' % (data_dir, cln))
    sdss_source_name = sdss['source_name'].values
    sdss_total_flux = sdss['int_flux'].values
    sdss_box = sdss['box'].values
    sdss16_name = sdss['SDSS16'].values

    panstarrs = pd.read_csv('%s/FIRST_HeTu_paper_%s_SDSS_DR16_PanSTARRS_DR1_diff.csv' % (data_dir, cln))
    panstarrs_source_name = panstarrs['source_name'].values
    panstarrs_total_flux = panstarrs['int_flux'].values
    panstarrs_box = panstarrs['box'].values
    panstarrs_ra = panstarrs['RAJ2000'].values
    panstarrs_dec = panstarrs['DEJ2000'].values
   
    print("processing %s ------" % cln)
    nn = 0
    ss = 0
    pp = 0
    for m in range(len(hetu_source_name)):
        yes = 0
        yes_s = 0
        yes_p = 0
        for n in range(len(ned_source_name)):
              if ned_num[n] == m:
                 ned_type_final.append(ned_type[n])
                 ned_name_final.append(ned_source_name[n])
                 redshift_final.append(ned_Redshift[n])
                 nn = nn + 1
                 print(nn)
                 yes = 1
                 break
        if yes == 0:
           print(hetu_source_name[m])
           ned_type_final.append('')
           ned_name_final.append('')
           redshift_final.append('--')
        
        for s in range(len(sdss_source_name)):
              if sdss_source_name[s] == hetu_source_name[m] and sdss_box[s] == hetu_box[m] and sdss_total_flux[s] == hetu_total_flux[m]:
                 sdss_name_final.append(sdss16_name[s])
                 yes_s = 1
                 ss = ss + 1       
                 print(ss)
                 break
        if yes_s == 0:
           sdss_name_final.append('')      

        for p in range(len(panstarrs_source_name)):
              if panstarrs_source_name[p] == hetu_source_name[m] and panstarrs_box[p] == hetu_box[m]: #and panstarrs_total_flux[p] == hetu_total_flux[m]:
                 panstarrs_ra_final.append(panstarrs_ra[p])
                 panstarrs_dec_final.append(panstarrs_dec[p])
                 yes_p = 1
                 pp = pp + 1
                 print('processing panstarrs %d' % pp)
                 break
        if yes_p == 0:
           panstarrs_ra_final.append('')
           panstarrs_dec_final.append('')



for cln in ['fr1']:#, 'fr2', 'ht', 'cj']: 
    hetu_table = pd.read_csv('%s/FIRST_HeTu_paper_%s.csv' % (data_dir, cln))
    hetu_table['SDSS16'] = sdss_name_final
    hetu_table['panstarrs_ra'] = panstarrs_ra_final
    hetu_table['panstarrs_dec'] = panstarrs_dec_final
    hetu_table['Type'] = ned_type_final
    hetu_table['NED_name'] = ned_name_final
    hetu_table['Redshift'] = redshift_final
    hetu_table.to_csv('%s/FIRST_HeTu_paper_%s_optical_ned.csv' % (data_dir, cln), index = False)
