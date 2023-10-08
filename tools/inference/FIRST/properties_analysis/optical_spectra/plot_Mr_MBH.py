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

Mrs = hetu_csv[hetu_csv['Mr'].notnull()]

mean1 =np.mean(Mrs['Mr'].values)
sigma1 = np.std(Mrs['Mr'].values)

print('mean Mr ->', np.mean(Mrs['Mr'].values))
print('median Mr ->', np.median(Mrs['Mr'].values))

plt.figure()
ax1 = plt.gca()
Mrs_final = Mrs['Mr'].values

print('Mr lower than -26 -> ', len(Mrs_final[np.where(Mrs_final<-26)]))
print('Mr lower than -20 -> ', len(Mrs_final[np.where(Mrs_final<-20)]))

bin_heights1, bin_borders1, _ = ax1.hist(Mrs_final, bins=20, facecolor='#87CEEB', align='mid', histtype='bar', edgecolor='k', linewidth=1, linestyle='-')


#bin_centers1 = bin_borders1[:-1] + np.diff(bin_borders1) / 2
#popt1, _ = curve_fit(gaussian, bin_centers1, bin_heights1, p0=[1., mean1, sigma1])
#print("spix fitted para: %s" % popt1)
#
#x_interval_for_fit1 = np.linspace(-1.1, 3.1, len(spixs['alpha'].values))
#ax1.plot(x_interval_for_fit1, gaussian(x_interval_for_fit1, *popt1), '--k', linewidth=2)

plt.xlabel(r'$M_{\rm r}$', fontsize=16)
#ax1.set_xticklabels([])
ax1.tick_params(labelsize=16)
#plt.yscale('log')
#ax1.set_xlim(-1.1,3.1)
#ax1.set_ylim(0,500000)
ax1.set_xticks([-28, -26, -24, -22, -20, -18, -16, -14])
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

plt.tight_layout()

#plt.savefig('spix.png', dpi=600,format="png")
plt.savefig('Mr.pdf')


MBH = hetu_csv[hetu_csv['log(M_BH)'].notnull()]
MBH_final = MBH['log(M_BH)'].values

print('MBH lower than 7.5 -> ', len(MBH_final[np.where(MBH_final<7.)]))
print('MBH lower than 9.5 -> ', len(MBH_final[np.where(MBH_final<9.5)]))

plt.figure(2)
ax2 = plt.gca()
bin_heights1, bin_borders1, _ = ax2.hist(MBH_final, bins=20, facecolor='#87CEEB', align='mid', histtype='bar', edgecolor='k', linewidth=1, linestyle='-')

#bin_centers1 = bin_borders1[:-1] + np.diff(bin_borders1) / 2
#popt1, _ = curve_fit(gaussian, bin_centers1, bin_heights1, p0=[1., mean1, sigma1])
#print("spix fitted para: %s" % popt1)
#
#x_interval_for_fit1 = np.linspace(-1.1, 3.1, len(spixs['alpha'].values))
#ax1.plot(x_interval_for_fit1, gaussian(x_interval_for_fit1, *popt1), '--k', linewidth=2)

plt.xlabel(r'log($M_{\rm BH}$) ($M_{\odot}$)', fontsize=16)
#ax1.set_xticklabels([])
ax2.tick_params(labelsize=16)
#plt.yscale('log')
#ax1.set_xlim(-1.1,3.1)
#ax1.set_ylim(0,500000)
ax2.set_ylabel('Number', fontsize=16)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(which='both', width=2)
ax2.tick_params(which='major', length=7)
ax2.tick_params(which='minor', length=4, color='k')

ax2.tick_params(axis="x", direction="in")
ax2.tick_params(axis="y", direction="in")
ax2.tick_params(which="minor", axis="x", direction="in")
ax2.tick_params(which="minor", axis="y", direction="in")

plt.tight_layout()

#plt.savefig('spix.png', dpi=600,format="png")
plt.savefig('MBH.pdf')

plt.figure(3)

Lrad = Mrs[Mrs['log(Lrad)'].notnull()]

Mrs_final = Lrad['Mr'].values
Lrad_final = Lrad['log(Lrad)'].values

ax3 = plt.gca()

ax3.plot(Mrs_final, Lrad_final, '+')

ax3.invert_xaxis()
ax3.set_xlabel(r'$M_{\rm r}$', fontsize=16)
ax3.set_ylabel(r'log($L_{\rm rad}$) (${\rm W} \cdot {\rm Hz}^{-1}$)', fontsize=16)
#ax1.set_xticklabels([])
ax3.tick_params(labelsize=16)
ax3.set_xticks([-14, -16, -18 , -20, -22, -24, -26, -28])
#plt.yscale('log')
#ax1.set_xlim(-1.1,3.1)
#ax1.set_ylim(0,500000)
#ax3.set_ylabel('Number', fontsize=16)
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.tick_params(which='both', width=2)
ax3.tick_params(which='major', length=7)
ax3.tick_params(which='minor', length=4, color='k')

ax3.tick_params(axis="x", direction="in")
ax3.tick_params(axis="y", direction="in")
ax3.tick_params(which="minor", axis="x", direction="in")
ax3.tick_params(which="minor", axis="y", direction="in")

plt.savefig('Mr_Lrad.pdf')
