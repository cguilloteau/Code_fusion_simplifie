# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:51:36 2019

@author: cguillot3

Ref 1 : C. Guilloteau, T. Oberlin, O. Berné, É. Habart, and N. Dobigeon
“Simulated JWST datasets for multispectral and hyperspectral image fusion”
The Astronomical Journal, vol. 160, no. 1, p. 28, Jun. 2020.

Ref 2 : C. Guilloteau, T. Oberlin, O. Berné, É. Habart, and N. Dobigeon
"Hyperspectral and Multispectral Image Fusion Under Spectrally Varying Spatial Blurs – Application to High Dimensional Infrared Astronomical Imaging"
IEEE Transactions on Computatonal Imaging, vol.6, Sept. 2020.

This code is used to store constants and paths for data fusion.
"""

"""
Paths for data and saving files
"""
DATA = 'Data/'
SAVE = 'Expe/'
SAVE2 = 'Expe/'

PSF = 'Data/PSF/'


"""
File name and path for hyperspectral and multispectral images
"""
MS_IM = DATA+'Ym_hst.fits'
HS_IM = DATA+'Yh_hst.fits'
# MS_IM = SAVE+'Ym_lowsnr_10.fits'
# HS_IM = SAVE+'Yh_lowsnr_10.fits'
# MS_IM = SAVE+'Ym_highsnr_psfsig-1.fits'
# HS_IM = SAVE+'Yh_highsnr_psfsig-1.fits'

"""
Subspace dimension
"""
lsub = 4

"""
Downsampling factor (integer)
"""
d = 3

"""
High spatial resolution image size with padding and padding factor
Example :
Original MS band size : 300x300
fact_pad : 41
nr = 300 + 2*fact_pad + 2 
nc = 300 + 2*fact_pad + 2
Conditions (to chose fact_pad) : nr % d = 0; nc % d = 0; (2*fact_pad+2) % d = 0; (fact_pad//d + 1) % 2 = 0
"""
nr = 144
nc = 144
fact_pad = 11

"""
Conversion constants : mJy/arcsec**2 to mJy/pixel (depends on the observation instruments)
"""
FLUXCONV_NC = 0.031**2
FLUXCONV_NS = 0.01

"""
*******************************************************************************************************
CONSTANTS FOR NOISE GENERATION, only used to generate/simulate HS and MS data
*******************************************************************************************************
"""

"""
Path for H2RG correlation matrix for noise generation (only used to generate/simulate HS and MS data)
"""
CORR = DATA+'h2rg_corr.fits'

"""
Constants for noise generation
"""
EXPTIME_NC = 128.841 # Exposure time in seconds for the MS image
SIG2READ_NC = 2*(16.2)**2/2 # Noise constant in e-/s for the MS image
# SIG2READ_NC = 29000
NFRAMES_NC = 2 # Number of frames during the MS image acquisition
EXPTIME_NS = 257.682 # Exposure time in seconds for the HS image
SIG2READ_NS = 1.6*(6/88)**2 # Noise constant in e-/s for the HS image
# SIG2READ_NS = 500
NFRAMES_NS = 5 # Number of frames during the HS image acquisition
GAIN_NC = 2 # Gain in e-/ph for MS instrument
GAIN_NS = 1 # Gain in e-/ph for HS instrument