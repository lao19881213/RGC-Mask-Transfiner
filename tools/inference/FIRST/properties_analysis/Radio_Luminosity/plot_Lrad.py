import numpy as np
import pandas as pd
import matplotlib
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

def gaussian(x, mean, amplitude, standard_deviation):
    return (amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2)))

FIRST_result = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FRIIRGcat_final.csv'

hetu_csv = pd.read_csv(FIRST_result)

Lrad_csv = hetu_csv[hetu_csv['log(Lrad)'].notnull()]
#z_csv = hetu_csv[hetu_csv['z'].notnull()]

plt.figure()
#fig,ax = plt.subplots(nrows=1,ncols=2,sharey='row')
ax1 = plt.gca()
#print(ax.shape)
#ax1 = ax[0]#[0]
#ax2 = ax[1]#[1]
xy = np.vstack([Lrad_csv['z'].values,Lrad_csv['log(Lrad)'].values])
density = gaussian_kde(xy)(xy)
density = density/(max(density)-min(density))

#plt.plot(Lrad_csv['z'].values, Lrad_csv['log(Lrad)'].values, 'o', markersize=2.0, markerfacecolor='none', markeredgecolor='m', alpha=0.3)

ax1.scatter(x=Lrad_csv['z'].values, y=Lrad_csv['log(Lrad)'].values, c=density, cmap='plasma')
y = np.linspace(22, 31, 100)
x = np.ones(len(y))*4
ax1.plot(x, y, 'r--')
print(np.mean(Lrad_csv['log(Lrad)'].values))
print(np.median(Lrad_csv['log(Lrad)'].values))
#plt.title('Luminosity of Radio Sources increasing with redshift')
ax1.set_ylabel(r'log($L_{\rm rad}$) (${\rm W} \cdot {\rm Hz}^{-1}$)', fontsize=16)
ax1.set_xlabel('Redshift ($z$)', fontsize=16)
#plt.semilogy()

#ticks_loc = ax1.get_yticks().tolist()
#print(ticks_loc, ax1.get_yticks().tolist())
#ax1.set_xticklabels([])
ax1.tick_params(labelsize=16)
#plt.yscale('log')
#ax1.set_xlim(-1.1,3.1)
ax1.set_ylim(22,31)
#ax1.set_ylabel('Number', fontsize=16)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(which='both', width=2)
ax1.tick_params(which='major', length=7)
ax1.tick_params(which='minor', length=4, color='k')

ax1.tick_params(axis="x", direction="in")
ax1.tick_params(axis="y", direction="in")
ax1.tick_params(which="minor", axis="x", direction="in")
ax1.tick_params(which="minor", axis="y", direction="in")
#plt.tick_params(labelsize=20)
#labels = ax1.get_xticklabels() + ax1.get_yticklabels()
#[label.set_fontname('Times New Roman') for label in labels]
#plt.tick_params(which='major', length=8,direction='in')
#plt.tick_params(which='minor', length=4,direction='in')
#ax1.tick_params(axis='both', which='both', width=1,direction='in')
#ax1.tick_params(axis = 'x', which = 'major', labelsize = 14, pad=4,direction='in')
#ax1.tick_params(axis = 'y', which = 'major', labelsize = 14, pad=4,direction='in')
#ax2.invert_yaxis()
#
#ax2.hist(Lrad_csv['log(Lrad)'].values, bins=20, facecolor='m', align='mid', histtype='bar', edgecolor='k', linewidth=2, linestyle='-', orientation='horizontal')
#
#ax2.set_xlabel('Number', fontsize=16)
#ax2.set_yticklabels([])
#ax2.xaxis.set_minor_locator(AutoMinorLocator())
#ax2.yaxis.set_minor_locator(AutoMinorLocator())
#ax2.tick_params(which='both', width=2)
#ax2.tick_params(which='major', length=7)
#ax2.tick_params(which='minor', length=4, color='k')
#
#ax2.tick_params(axis="x", direction="in")
#ax2.tick_params(axis="y", direction="in")
#ax2.tick_params(which="minor", axis="x", direction="in")
#ax2.tick_params(which="minor", axis="y", direction="in")

#colorbar = plt.colorbar()
#colorbar.set_label('normalized density')

plt.tight_layout()

#plt.savefig('spix.png', dpi=600,format="png")
plt.savefig('Lrad.pdf')

