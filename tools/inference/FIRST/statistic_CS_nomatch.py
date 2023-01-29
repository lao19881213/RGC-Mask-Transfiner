#!/usr/bin/env python


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


data_dir = "/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results"


nomatch=pd.read_csv(os.path.join(data_dir,'nomatch_hetu_final.csv'),sep=',')

nomatch1=nomatch[nomatch['peak_flux']!=np.inf]

peak = nomatch['peak_flux'].values
local_rms = nomatch['local_rms'].values
SNR = peak / local_rms

print(len(SNR[SNR<=5]))

print(np.min(SNR))
plt.figure()
ax = plt.gca()
#plt.hist(offset_x, bins=33, label=r'$\Delta x$', color='b')
plt.hist(SNR,bins=10000, facecolor='#1F77B4', align='mid', histtype='bar', edgecolor='r', linewidth=0.2, linestyle='-',alpha=0.5)
#ax.hist(offset_x,bins=33, facecolor='#FF7F0E', align='mid', histtype='bar', edgecolor='k', linewidth=0.2, linestyle='-',alpha=0.5, label=r'$\Delta y$')
plt.xlabel(r'S/N', fontsize=18)
plt.tick_params(labelsize=16)
#plt.yscale('log')
plt.xlim(0,200)
plt.ylabel('Number of sources', fontsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4, color='k')

plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.tick_params(which="minor", axis="x", direction="in")
plt.tick_params(which="minor", axis="y", direction="in")
#plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig('snr_nomatch.png', dpi=600,format="png",bbox_inches = 'tight')
plt.savefig('snr_nomatch.pdf', dpi=300,format="pdf")




