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

Modify paths and constants in the 'CONSTANTS.py' file, if necessary.
Run this code to test fusion.
"""

import fusion
import errors
import sys
from CONSTANTS import *
import numpy as np
from astropy.io import fits
# from sparse_preprocess import get_hsms, get_throughput, set_inputs
from scipy import linalg
import tools

import warnings
warnings.filterwarnings('ignore')


def main(args):
    return fusion.fusion_reginf(lsub, MS_IM, HS_IM)


# def choose_subspace(args):
#     """
#     Plots and saves PCA eigenvalues of the covariance matrix to choose the spectral subspace dimension.
#     """
#     #### Preprocessing of HS and MS images, operators and regularization
#     Ym, Yh, tabwave, sig2 = get_hsms(MS_IM, HS_IM)
#     Lh = get_throughput(tabwave, 'f170lp', 'g235h')
#     # Perform PCA on the HS image
#     # Reshapes
#     l, m, n = Yns.shape
#     Yh_ = np.reshape(np.dot(np.diag(Lh**-1), np.reshape(Yh, (l, m*n))), (l, m, n))
#     # PCA
#     L_h, S_hx, S_hy = Yh_.shape
#     X = np.reshape(Yh_.copy(), (L_h, S_hx*S_hy)).T
#     L, M = X.shape
#     X_mean = np.mean(X, axis=0)
#     X -= X_mean
#     U, S, V = linalg.svd(X, full_matrices=False)
#     plt.semilogy(S)
#     plt.title('Eigenvalues -- PCA')
#     plt.savefig(SAVE+'eigenvalues_pca.png')
#     return S


# def check_operators(args):
#     # Get images
#     Ym, Yh, Lm, Lh, V, Z, D, Wd, sig2 = set_inputs(lsub, nr, nc, MS_IM, HS_IM, -1)

#     # VZ single band
#     l = 100
#     Zifft = np.reshape(np.fft.ifft2(np.reshape(Z, (lsub, nr, nc)), norm='ortho'), ((lsub, nr*nc)))
#     band = np.reshape(np.dot(V[l],Zifft), (nr,nc))
#     bandfft = np.reshape(np.dot(V[l],np.reshape(Z, (lsub, nr*nc))), (nr, nc))
#     plt.imshow(np.real(band))
#     plt.colorbar()
#     plt.savefig(SAVE+'VZ_'+str(l)+'.png')
#     plt.close()

#     # Convolution operators
#     mvz = np.reshape(tools.get_h_band(l), (nr, nc))*bandfft
#     mvz = np.fft.ifft2(mvz, norm='ortho')
#     hvz = np.reshape(tools.get_g_band(l), (nr, nc))*bandfft
#     hvz = np.fft.ifft2(hvz, norm='ortho')
#     plt.imshow(mvz)
#     plt.colorbar()
#     plt.savefig(SAVE+'M(VZ)'+str(l)+'.png')
#     plt.close()
#     plt.imshow(hvz)
#     plt.colorbar()
#     plt.savefig(SAVE+'H(VZ)_'+str(l)+'.png')
#     plt.close()

#     # Spectral degradation
#     LmV = np.dot(Lm,V)
#     LmVZ = np.reshape(np.dot(LmV[3],Zifft), (nr,nc))
#     plt.imshow(np.real(LmVZ))
#     plt.colorbar()
#     plt.savefig(SAVE+'LmVZ_'+str(l)+'.png')
#     plt.close()


if __name__ == "__main__":
    image = main(sys.argv)
    # eigen = choose_subspace(sys.argv)
    # check_operators(sys.argv)

