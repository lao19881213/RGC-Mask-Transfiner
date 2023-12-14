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

from statistics import mean
from scipy.optimize import leastsq

def func(p, x):
    k, b = p
    return k*x+b

def error(p, x, y):
    return func(p, x)-y

FIRST_result = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FRIIRGcat_final.csv'

hetu_csv = pd.read_csv(FIRST_result)

OIII = hetu_csv[hetu_csv['log(L_[OIII])'].notnull()]
OIII_rad = OIII[OIII['log(Lrad)'].notnull()]

OIII_rad_class = OIII_rad[OIII_rad['class'].notnull()]

Lrad = OIII_rad_class['log(Lrad)'].values
L_OIII = OIII_rad_class['log(L_[OIII])'].values
class_hl = OIII_rad_class['class'].values

plt.figure()
ax1 = plt.gca()
class_H_Lrad = []
class_L_Lrad = []
class_H_LOIII = []
class_L_LOIII = []
#10^7 erg s^-1 = 1 W
for mm in range(len(OIII_rad_class['class'].values)):
    if class_hl[mm] == 'LERG':
       plt.plot(Lrad[mm]+7, L_OIII[mm], 'o', markerfacecolor='None', markeredgecolor='g', zorder=0)#, markersize=4)
       class_L_Lrad.append(Lrad[mm])
       class_L_LOIII.append(L_OIII[mm])
    elif class_hl[mm] == 'HERG':
       plt.plot(Lrad[mm]+7, L_OIII[mm], 'o', markerfacecolor='None', markeredgecolor='m', zorder=1)#, markersize=4)
       class_H_Lrad.append(Lrad[mm])
       class_H_LOIII.append(L_OIII[mm])

p0 = [1, 20]

X = np.array(class_H_Lrad) + 7
Y = np.array(class_H_LOIII)

Para = leastsq(error, p0, args=(X, Y))

k, b = Para[0]
print("k=", k, "b=", b)

x = np.linspace(29.0, 36.0, 100)  #
y = k*x+b  # 
ax1.plot(x, y, "b--", label="NH", linewidth=2)

X1 = np.array(class_L_Lrad) + 7
Y1 = np.array(class_L_LOIII)

Para1 = leastsq(error, p0, args=(X1, Y1))

k1, b1 = Para1[0]
print("k1=", k1, "b1=", b1)

x1 = np.linspace(29.0, 36.0, 100)  #
y1 = k1*x1+b1  # 
ax1.plot(x1, y1, "k--", label="NH1", linewidth=2)

ticks_loc = ax1.get_xticks().tolist()
print(ticks_loc, ax1.get_xticks().tolist())

#y = np.linspace(-1.7, 4.3, 100)
#x = np.ones(len(y))*2.6
#ax1.plot(x, y, 'k--', linewidth =0.5)
#
#
#x = np.linspace(0.96, 5.1, 100)
#y = np.ones(len(x))*1.7
#ax1.plot(x, y, 'k--', linewidth =0.5)
#
#x = np.linspace(0.96, 5.1, 100)
#y = np.ones(len(x))*1.45
#ax1.plot(x, y, 'k--', linewidth =0.5)
#
#plt.ylabel(r'Dn4000', fontsize=16)
#ax1.set_xticklabels([])
ax1.tick_params(labelsize=16)
#plt.yscale('log')
ax1.set_xlim(29.5,36)
#ax1.set_ylim(-1.7,4.3)
#ax1.set_xticks([-28, -26, -24, -22, -20, -18, -16, -14])
ax1.set_xlabel(r'log($L_{\rm rad}$) (${\rm erg} \cdot {\rm s}^{-1} \cdot {\rm Hz}^{-1}$)', fontsize=16)
ax1.set_ylabel(r'log($L_{\rm [OIII]}$) (${\rm erg} \cdot {\rm s}^{-1}$)', fontsize=16)
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
plt.savefig('Lrad_Loiii.pdf')
