import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

result_file = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_flux_fixed_vlass.csv'
hetu_csv = pd.read_csv(result_file)
RPAs = hetu_csv['RPA'].values
ras = hetu_csv['centre_ra'].values
decs = hetu_csv['centre_dec'].values

RPAs_N = []
RPAs_S = []
for mm in range(len(RPAs)):
    #ra: 6h-18h, 90d-270d
    if ras[mm] >= 90 and ras[mm] <= 270:
       RPAs_N.append(RPAs[mm])
    else:
       RPAs_S.append(RPAs[mm])

fig = plt.figure()
ax1 = plt.gca()
#ax1.hist(RPAs, bins=20, facecolor='k', align='mid', histtype='step', edgecolor='k', linewidth=2, linestyle='-')
#ax1.hist(RPAs_N, bins=20, facecolor='r', align='mid', histtype='step', edgecolor='r', linewidth=2, linestyle='-')
ax1.hist(RPAs_S, bins=20, facecolor='b', align='mid', histtype='step', edgecolor='b', linewidth=2, linestyle='-')
plt.tick_params(labelsize=16)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.tick_params(which='major', length=8,direction='in')
plt.tick_params(which='minor', length=4,direction='in')
ax1.tick_params(axis='both', which='both', width=1,direction='in')
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14, pad=4,direction='in')
ax1.tick_params(axis = 'y', which = 'major', labelsize = 14, pad=4,direction='in')
ax1.set_ylabel('Number', fontsize=20) 
ax1.set_xlabel('Position angle (degrees)', fontsize=20)
plt.tight_layout()
plt.savefig('RPAs.pdf')
