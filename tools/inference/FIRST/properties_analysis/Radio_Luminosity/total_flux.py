
import pandas as pd


for cln in ['fr1', 'fr2', 'ht', 'cj']:
    ned_table = pd.read_csv('../NED/centre_peaks_5arcsec/info_%s.csv' % cln)
    hetu_num = ned_table['HeTu_num'].values
    hetu = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper.csv')
    total_flux = hetu['int_flux'].values

    total_flux_ned = []
    for n in hetu_num:
        total_flux_ned.append(total_flux[n])
    ned_table['total_flux'] = total_flux_ned
    ned_table.to_csv('../NED/centre_peaks_5arcsec/info_%s_flux.csv' % cln, index = False) 

