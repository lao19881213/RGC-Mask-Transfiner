import pandas as pd
import csv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--inpfn', dest='inpfn', type=str, default='/o9000/ASKAP/AS111/hetu_results/D1v2_gwc_SB10463_snr10_1.csv', help='pred input result file')
parser.add_argument('--outdir', dest='outdir', type=str, default='/o9000/ASKAP/AS111/hetu_result', help='output png file directory')
args = parser.parse_args()

cmd_str = "topcat -stilts tmatch1 matcher=sky params=5 values='centre_ra centre_dec' in='%s' ifmt=csv action=identify omode=out out='%s_matched.csv' ofmt=csv" % (args.inpfn, os.path.splitext(args.inpfn)[0]) # 5 arcsec
print(cmd_str)
os.system(cmd_str)

frame=pd.read_csv('%s_matched.csv' % os.path.splitext(args.inpfn)[0])
data = frame.drop_duplicates(subset=['label','GroupID','GroupSize'], keep='first', inplace=False)
data.to_csv('%s_final.csv' % os.path.splitext(args.inpfn)[0], index = False)#, encoding='utf8')

df = pd.read_csv('%s_matched.csv' % os.path.splitext(args.inpfn)[0])
#print(df)
data_new = df[(df.GroupID.isnull()) & (df.GroupSize.isnull())]
print(data_new)
data_new.to_csv('%s_final.csv' % os.path.splitext(args.inpfn)[0], mode='a', header=False, index = False)

