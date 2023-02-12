#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

hetu=pd.read_csv(os.path.join(data_dir,'FIRST_HeTu.csv'),sep=',')

hetu_cs = hetu[hetu['label']=='cs']
hetu_fr1 = hetu[hetu['label']=='fr1']
hetu_fr2 = hetu[hetu['label']=='fr2']
hetu_ht = hetu[hetu['label']=='ht']
hetu_cj = hetu[hetu['label']=='cj']

print(hetu_cs.shape[0])
print(hetu_fr1.shape[0])
print(hetu_fr2.shape[0])
print(hetu_ht.shape[0])
print(hetu_cj.shape[0])



print(np.mean(hetu_cs['score'].values))
print(np.mean(hetu_fr1['score'].values))
print(np.mean(hetu_fr2['score'].values))
print(np.mean(hetu_ht['score'].values))
print(np.mean(hetu_cj['score'].values))

plt.figure()
ax = plt.gca()
#plt.hist(offset_x, bins=33, label=r'$\Delta x$', color='b')
#plt.hist(hetu_cs['score'].values,bins=35, facecolor='k', align='mid', histtype='step', edgecolor='k', linewidth=1.0, linestyle='-', label='CS')
plt.hist(hetu_fr1['score'].values,bins=20, facecolor='r', align='mid', histtype='step', edgecolor='r', linewidth=1.0, linestyle='-', label='FRI')
plt.hist(hetu_fr2['score'].values,bins=20, facecolor='b', align='mid', histtype='step', edgecolor='b', linewidth=1.0, linestyle='-', label='FRII')
plt.hist(hetu_ht['score'].values,bins=20, facecolor='g', align='mid', histtype='step', edgecolor='g', linewidth=1.0, linestyle='-', label='HT')
plt.hist(hetu_cj['score'].values,bins=20, facecolor='m', align='mid', histtype='step', edgecolor='m', linewidth=1.0, linestyle='-', label='CJ')
plt.xlabel(r'Prediction score', fontsize=18)
plt.tick_params(labelsize=16)
plt.yscale('log')
plt.xlim(0.45,1)
print(ax.get_ylim()[1])
#plt.ylim(0.1,ax.get_ylim()[1])
plt.ylabel('Number of sources', fontsize=18)
#ax.xaxis.set_minor_locator(AutoMinorLocator())
##ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.tick_params(which='both', width=2)
#ax.tick_params(which='major', length=7)
#ax.tick_params(which='minor', length=4, color='k')
#
#plt.tick_params(axis="x", direction="in")
#plt.tick_params(axis="y", direction="in")
#plt.tick_params(which="minor", axis="x", direction="in")
#plt.tick_params(which="minor", axis="y", direction="in")
plt.tick_params(labelsize=20)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.tick_params(which='major', length=8,direction='in')
plt.tick_params(which='minor', length=4,direction='in')
ax.tick_params(axis='both', which='both', width=1,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 14, pad=4,direction='in')
ax.tick_params(axis = 'y', which = 'major', labelsize = 14, pad=4,direction='in')
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('mask_accuracy.png', dpi=600,format="png",bbox_inches = 'tight')
plt.savefig('mask_accuracy.pdf', dpi=300,format="pdf")


