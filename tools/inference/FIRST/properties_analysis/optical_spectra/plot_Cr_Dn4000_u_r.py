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

FIRST_result = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FRIIRGcat_final.csv'

hetu_csv = pd.read_csv(FIRST_result)

Crs = hetu_csv[hetu_csv['Cr'].notnull()]

Dn4000 = Crs[Crs['Dn4000'].notnull()]

class_hl = Dn4000['class'].values
Crs = Dn4000['Cr'].values
Dn4000s = Dn4000['Dn4000'].values
print(np.mean(Dn4000s))
class_hl_L = Dn4000[Dn4000['class']=='LERG']
print('len LERGs ->',len(class_hl_L))
class_hl_L_1 = class_hl_L[class_hl_L['Dn4000']>=1.7]
print('Cr > = 2.6 and Dn4000 > 1.7 ->', len(class_hl_L_1[class_hl_L_1['Cr']>=2.6]))
#class_hl_L_2 = class_hl_L_1[class_hl_L_1['z']<=0.15]
#print('red galaxies ->', len(class_hl_L_2[class_hl_L_2['z']>=0.1]))
#class_hl_L_3 = class_hl_L_1[class_hl_L_1['Cr']>=2.6]
#class_hl_L_4 = class_hl_L_3[class_hl_L_3['z']<=0.15]
#print('red ETGs ->', len(class_hl_L_4[class_hl_L_4['z']>=0.1]))

print(len(Dn4000['Cr'].values))

class_hl_H = Dn4000[Dn4000['class']=='HERG']
print('len HERGs ->', len(class_hl_H))
class_hl_H_1 = class_hl_H[class_hl_H['Dn4000']<=1.45]
print('len Dn4000 <=1.45 ->', len(class_hl_H_1))#
print('len Dn4000 <=1.45 and Cr >= 2.6 ->',len(class_hl_H_1[class_hl_H_1['Cr']>=2.6]))

plt.figure()
ax1 = plt.gca()

for mm in range(len(Dn4000['Cr'].values)):
    if class_hl[mm] == 'LERG':
       plt.plot(Crs[mm], Dn4000s[mm], 'o', markerfacecolor='None', markeredgecolor='g', zorder=0)#, markersize=4)
    elif class_hl[mm] == 'HERG':
       plt.plot(Crs[mm], Dn4000s[mm], 'o', markerfacecolor='None', markeredgecolor='m', zorder=1)#, markersize=4)
    #else:
    #   print('no HL')
       #plt.plot(Crs[mm], Dn4000s[mm], 'o', markerfacecolor='None', markeredgecolor='k',  alpha=0.5)
#ax1.plot(x_interval_for_fit1, gaussian(x_interval_for_fit1, *popt1), '--k', linewidth=2)

ticks_loc = ax1.get_xticks().tolist()
print(ticks_loc, ax1.get_xticks().tolist())

y = np.linspace(-1.7, 4.3, 100)
x = np.ones(len(y))*2.6
ax1.plot(x, y, 'k--', linewidth =0.5)


x = np.linspace(0.96, 5.1, 100)
y = np.ones(len(x))*1.7
ax1.plot(x, y, 'k--', linewidth =0.5)

x = np.linspace(0.96, 5.1, 100)
y = np.ones(len(x))*1.45
ax1.plot(x, y, 'k--', linewidth =0.5)

plt.xlabel(r'$C_{\rm r}$', fontsize=16)
plt.ylabel(r'Dn4000', fontsize=16)
#ax1.set_xticklabels([])
ax1.tick_params(labelsize=16)
#plt.yscale('log')
ax1.set_xlim(0.96,5.1)
ax1.set_ylim(-1.7,4.3)
#ax1.set_xticks([-28, -26, -24, -22, -20, -18, -16, -14])
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

plt.tight_layout()

#plt.savefig('spix.png', dpi=600,format="png")
plt.savefig('Cr_Dn4000.pdf')

HL = hetu_csv[hetu_csv['class']!='']

u_r = HL[HL['u-r'].notnull()]
u_r_Mr = u_r[u_r['Mr'].notnull()]
u_r_ETGs = u_r_Mr[u_r_Mr['Cr']>=2.6]

plt.figure()
ax2 = plt.gca()

class_HL = u_r_ETGs['class'].values
Mr_HL = u_r_ETGs['Mr'].values
u_r_HL = u_r_ETGs['u-r'].values
Mr_final = []
u_r_final = []
Mr_final_L = []
u_r_final_L = []

for mm in range(len(class_HL)):
    if class_HL[mm] == 'LERG':
       plt.plot(Mr_HL[mm], u_r_HL[mm], 'o', markerfacecolor='None', markeredgecolor='g', zorder=0)
       Mr_final.append(Mr_HL[mm])
       u_r_final.append(u_r_HL[mm])
       Mr_final_L.append(Mr_HL[mm])
       u_r_final_L.append(u_r_HL[mm])
    elif class_HL[mm] == 'HERG':
       plt.plot(Mr_HL[mm], u_r_HL[mm], 'o', markerfacecolor='None', markeredgecolor='m', zorder=1)
       Mr_final.append(Mr_HL[mm])
       u_r_final.append(u_r_HL[mm])

    #else:
    #   print('no HL')
       #plt.plot(Crs[mm], Dn4000s[mm], 'o', markerfacecolor='None', markeredgecolor='k',  alpha=0.5)
#ax1.plot(x_interval_for_fit1, gaussian(x_interval_for_fit1, *popt1), '--k', linewidth=2)

ticks_loc = ax2.get_xticks().tolist()
print(ticks_loc, ax2.get_xticks().tolist())
#
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
#plt.xlabel(r'$C_{\rm r}$', fontsize=16)
#plt.ylabel(r'Dn4000', fontsize=16)
##ax1.set_xticklabels([])
k = -0.18357506505429885 
b = -0.8455068727238104 - 0.6

x = np.linspace(-18.6, -24.6, 100)
y = k*x + b
ax2.plot(x, y, 'b--', linewidth =0.5)

cnt = 0
for nn in range(len(Mr_final_L)):
    if u_r_final_L[nn] < k*Mr_final_L[nn] + b:
       cnt = cnt + 1

print('blue FR-II LERGs ->', cnt) 

ax2.tick_params(labelsize=16)
##plt.yscale('log')
#ax2.set_xlim(-18,-25)
#ax1.set_ylim(-1.7,4.3)
#ax1.set_xticks([-28, -26, -24, -22, -20, -18, -16, -14])
#ax1.set_ylabel('Number', fontsize=16)
ax2.invert_xaxis()
ax2.set_xlim(-24.6,-18.6)
ax2.invert_xaxis()
ax2.set_xlabel(r'$M_{\rm r}$', fontsize=16)
ax2.set_ylabel(r'u-r color', fontsize=16)
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
plt.savefig('Mr_ur.pdf')
#plt.clf()


#u_r_Mr_final = u_r_ETGs[u_r_ETGs['class']!='']
#Mr_HL = u_r_Mr_final['Mr'].values
#u_r_HL = u_r_Mr_final['u-r'].values
plt.figure(3)
ax3 = plt.gca()
xy = np.vstack([Mr_final, u_r_final])
#print(Mr_final)
#print(u_r_final)
density = gaussian_kde(xy)(xy)
density = density/(max(density)-min(density))

Mr_final_best = []
u_r_final_best = []
for mm in range(len(Mr_final)):
    if u_r_final[mm] >=2.4 and u_r_final[mm] <=4.0:
       Mr_final_best.append(Mr_final[mm])
       u_r_final_best.append(u_r_final[mm])

def best_fit_slope_and_intercept(xs, ys):
       m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
       b = mean(ys) - m * mean(xs)
       return m, b

m, b = best_fit_slope_and_intercept(np.array(Mr_final_best), np.array(u_r_final_best))

regression_line = []
for a in Mr_final:
    regression_line.append((m * a) + b)

print(m,b)
ax3.plot(Mr_final, regression_line, 'r--', lw=0.8)
ax3.plot(Mr_final, np.array(regression_line)-0.6, 'r--', lw=0.8)
print(np.where(density==np.max(density)))

#print(Mr_final[1451])
#print(u_r_final[1451])
ax3.scatter(x=Mr_final, y=u_r_final, c=density, cmap='plasma')
ax3.invert_xaxis()
plt.savefig('Mr_ur_dens.pdf')
