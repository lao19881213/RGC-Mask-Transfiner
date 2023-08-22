

#Inputs:
z=0.0       #redshift
alpha_arcsec=1.0   #angular size of source in arcsec


#Importing the library packages:
import os,sys,shutil,re
import numpy as np
import pandas as pd


#Function to estimate cosmological scale in kpc/arcsec or pc/mas unit for the given redshift of source:
# The output of this function can be verified using the Cosmology calculator webpage:
#    http://www.astro.ucla.edu/~wright/CosmoCalc
# The cosmological parameters taken here are generally used ones. If you know the cosmological parameters for your case, use them in the following function for better accuracy.
def get_scale(z):
    # Cosmological parameters:
    H0 = 70         # Hubble constant in units of km/s/Mpc
    WM = 0.3        # Omega(matter)
    universe='flat' # Assuming flat Universe cosmology
    # Omega(vacuum) or lambda is specified according to the Universe type:
    if universe=='open':
        WV = 0.0
    elif universe=='flat':
        WV = 1.0 - WM
    else:
        WV = 1.0 - WM - 0.4165/(H0*H0)   # Universe in general or user specified value

    # Initialize constants:
    WR = 0.         # Omega(radiation)
    WK = 0.         # Omega curvature = 1-Omega(total)
    c = 299792.458  # velocity of light in km/sec
    DCMR = 0.0      # comoving radial distance in units of c/H0
    DA = 0.0        # angular size distance
    DA_Mpc = 0.0
    kpc_DA = 0.0
    a = 1.0         # 1/(1+z), the scale factor of the Universe
    az = 0.5        # 1/(1+z(source))
    h = H0/100.
    WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1-WM-WR-WV
    az = 1.0/(1+1.0*z)
    n=1000          # number of points in integrals
    DCMR = 0.0

    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
        a = az+(1-az)*(i+0.5)/n
        adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        DCMR = DCMR + 1./(a*adot)
    DCMR = (1.-az)*DCMR/n

    # tangential comoving distance
    ratio = 1.00
    x = np.sqrt(abs(WK))*DCMR
    if x > 0.1:
        if WK > 0:
            ratio =  0.5*(exp(x)-exp(-x))/x
        else:
            ratio = sin(x)/x
    else:
        y = x*x
    if WK < 0: y = -y
    ratio = 1. + y/6. + y*y/120.
    DCMT = ratio*DCMR
    DA = az*DCMT
    DA_Mpc = (c/H0)*DA
    kpc_DA = DA_Mpc/206.264806    #scale in kpc/arcsec or pc/mas

    return kpc_DA


if z!=0.0:
    scale=get_scale(z)
    #This is cosmological scale conversion factor which can be used to convert angular distance into linear distance.
    d_pc = scale * alpha_arcsec

