import numpy as np
import os
import pandas as pd
import csv

datadir = "/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results"

for cls in ['fr2', 'ht', 'cj']:
    sdss_csv_f = "FIRST_HeTu_paper_%s_SDSS_DR16.csv" % cls
    panstarrs_csv_f = "FIRST_HeTu_paper_%s_PanSTARRS_DR1.csv" % cls
    sdss_csv = pd.read_csv(os.path.join(datadir, sdss_csv_f))
    
    panstarrs_csv = pd.read_csv(os.path.join(datadir, panstarrs_csv_f))
    
    diff = pd.concat([sdss_csv, panstarrs_csv]).drop_duplicates(['source_name'], keep=False)
    
    diff.to_csv(os.path.join(datadir, "FIRST_HeTu_paper_%s_SDSS_DR16_PanSTARRS_DR1_diff.csv" % cls), index=False)
