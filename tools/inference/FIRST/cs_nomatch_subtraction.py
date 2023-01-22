import numpy as np
import os
import pandas as pd
import csv

datadir = "/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results"

cs_csv_f = "FIRST_infer_part0-4_th0.1_cs_final_fixed.csv"
match_csv_f = "FIRST_infer_part0-4_th0.1_first_14dec17_matched_cs_nocentre_final.csv"
cs_csv = pd.read_csv(os.path.join(datadir, cs_csv_f))#, usecols=['uuid', 'ra_170_231MHz', 'dec_170_231MHz'])

match_csv = pd.read_csv(os.path.join(datadir, match_csv_f))

diff = pd.concat([cs_csv, match_csv]).drop_duplicates(['objectname', 'imagefilename'], keep=False)

diff.to_csv(os.path.join(datadir, "nomatch_hetu.csv"), index=False)

cs_csv = pd.read_csv(os.path.join(datadir, 'nomatch_hetu.csv'))#, usecols=['uuid', 'ra_170_231MHz', 'dec_170_231MHz'])

match_csv = pd.read_csv(os.path.join(datadir, 'nomatch_hetu_match.csv'))

diff = pd.concat([cs_csv, match_csv]).drop_duplicates(['objectname', 'imagefilename'], keep=False)

diff.to_csv(os.path.join(datadir, "nomatch_hetu_final.csv"), index=False)

