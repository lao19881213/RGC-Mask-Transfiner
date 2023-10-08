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


def gaussian(x, mean, amplitude, standard_deviation):
    return (amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2)))

FIRST_result = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FRIIRGcat_final.csv'

hetu_csv = pd.read_csv(FIRST_result)

spixs = hetu_csv[hetu_csv['alpha'].notnull()]

mean1 =np.mean(spixs['alpha'].values)
sigma1 = np.std(spixs['alpha'].values)

print('mean spix ->', np.mean(spixs['alpha'].values))
print('median spix ->', np.median(spixs['alpha'].values))

spixs_lower = hetu_csv[hetu_csv['alpha']<0.0]
print(np.mean(spixs_lower['LAS']))

plt.figure()
ax1 = plt.gca()
spixs_final = spixs['alpha'].values
print('spix total -> ', len(spixs['alpha'].values))
print('spix greater than 0.5 -> ', len(spixs_final[np.where(spixs_final>0.5)]))
print('spix greater than 0.0 -> ', len(spixs_final[np.where(spixs_final>=0.0)]))
bin_heights1, bin_borders1, _ = ax1.hist(spixs['alpha'].values, bins=20, facecolor='m', align='mid', histtype='bar', edgecolor='k', linewidth=1, linestyle='-')

bin_centers1 = bin_borders1[:-1] + np.diff(bin_borders1) / 2
popt1, _ = curve_fit(gaussian, bin_centers1, bin_heights1, p0=[1., mean1, sigma1])
print("spix fitted para: %s" % popt1)

x_interval_for_fit1 = np.linspace(-1.1, 3.1, len(spixs['alpha'].values))
ax1.plot(x_interval_for_fit1, gaussian(x_interval_for_fit1, *popt1), '--k', linewidth=2)

plt.xlabel(r'Spectral index ($\alpha_{1.4}^3$)', fontsize=16)
#ax1.set_xticklabels([])
ax1.tick_params(labelsize=16)
#plt.yscale('log')
ax1.set_xlim(-1.1,3.1)
#ax1.set_ylim(0,500000)
ax1.set_ylabel('Number', fontsize=16)
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

plt.tight_layout()

#plt.savefig('spix.png', dpi=600,format="png")
plt.savefig('spix.pdf')


