import numpy as np
import os
import pandas as pd
import csv

datadir = "/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results"

centre_csv_f = "FIRST_infer_part0-4_th0.1_first_14dec17_matched_cs_centre.csv"
nocentre_csv_f = "FIRST_infer_part0-4_th0.1_first_14dec17_matched_cs_nocentre.csv"
centre_csv = pd.read_csv(os.path.join(datadir, centre_csv_f))#, usecols=['uuid', 'ra_170_231MHz', 'dec_170_231MHz'])

nocentre_csv = pd.read_csv(os.path.join(datadir, nocentre_csv_f))

diff = pd.concat([centre_csv, nocentre_csv]).drop_duplicates(['objectname'], keep=False)

diff.to_csv(os.path.join(datadir, "FIRST_infer_part0-4_th0.1_first_14dec17_matched_cs_diff.csv"), index=False)
