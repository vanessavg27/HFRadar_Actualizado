#!/usr/bin/env python3

import math,time,os,sys

import matplotlib.pyplot as plt
#mpl.rc('text', usetex = False)
from matplotlib import cm
import numpy as np
import scipy.constants as const
from scipy.signal import medfilt, find_peaks
from scipy.ndimage import median_filter as med_filt
from scipy.signal import savgol_filter as savgol
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev
from time import time
#from datetime import datetime
#import datetime
import os,sys
import h5py
import gc
from scipy.stats import median_absolute_deviation as MAD
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import wasserstein_distance as WassDistance
from scipy.ndimage import gaussian_filter1d as gaussfilt
from skimage.filters import  threshold_local
from scipy import sparse
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from scipy.sparse import csr_matrix
import argparse
import glob
#import datetime

def epsilon(data, vecinos = 10):

    neigh = NearestNeighbors(n_neighbors = vecinos)
    nbrs = neigh.fit(data)
    distancias, indices = nbrs.kneighbors(data)
    distancias = np.sort(distancias,axis=0)
    distancias = distancias[:,vecinos -1]
    distancias = sorted(distancias)
    i = np.arange(len(distancias))
    knee = KneeLocator(i, distancias, curve='convex', direction='increasing', interp_method='polynomial',online= True,S=1)
    eps=distancias[knee.knee]
    
    if eps > 10.5:
        eps = 5
        print("EPS EDITADO CON KNEED:",eps)
    elif eps<3:
        eps= 4.2
        print("EPS EDITADO CON KNEED:",eps)
    else:
        print("EPS CALCULADO CON KNEED:",eps)
    return eps

def delete_bands(ancho,matrix):
    Freqs = np.linspace(-5,5,matrix.shape[1])
    #NewFullSpectra = FullSpectra
    x_min = -ancho/2
    x_max =  ancho/2
    #print(Freqs[0])
    index_min = int(round((x_min-Freqs[0])/(Freqs[1]-Freqs[0])))
    index_max = int(round((x_max-Freqs[0])/(Freqs[1]-Freqs[0])))
    #min_value = np.min(FullSpectra)
    #print("index_min",index_min,type(index_min),"index_max",index_max,type(index_max))
    NewFullSpectra = np.zeros((matrix.shape[0],matrix.shape[1]))
    NewFullSpectra[:,index_min:index_max] = matrix[:,index_min:index_max]

    return NewFullSpectra

def ploteado(New_Full_Spectra_a,min_a,New_Full_Spectra_b,min_b,FullSpectra_a,FullSpectra_b,name,graphics_folder, eps_a,eps_b):
    import datetime
    tiempo = str(datetime.datetime.fromtimestamp(int(name)))
    print("Graphics_folder",graphics_folder)
    FS_filtrado_A = New_Full_Spectra_a.toarray()
    FS_filtrado_A[FS_filtrado_A == 0] = min_a


    FS_filtrado_B = New_Full_Spectra_b.toarray()
    FS_filtrado_B[FS_filtrado_B == 0] = min_b

    #pw_ch0 = np.fft.ifftshift(FS_filtrado_A)
    #pw_ch1 = np.fft.ifftshift(FS_filtrado_B)
    pw_ch0 = np.fft.ifftshift(FS_filtrado_A.T)
    pw_ch1 = np.fft.ifftshift(FS_filtrado_B.T)

    plt.figure(figsize=(20,4))
    plt.subplot(1,4,1)
    plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_a), cmap='jet')
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.title("%s - %s"%(tiempo,"CH0-ori"))
      
    plt.subplot(1,4,2)
    plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch0).T), cmap='jet')
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.title("%s - %s. EPS:%s"%(tiempo,"CH0-Filt",str(eps_a)))
    plt.subplot(1,4,3)
    plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_b), cmap='jet')
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.title("%s - %s"%(tiempo,"CH1-ori"))
    plt.subplot(1,4,4)
    plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch1).T), cmap='jet')
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.title("%s - %s. EPS:%s"%(tiempo,"CH1-Filt",str(eps_b)))
    try:
        plt.savefig(fname=graphics_folder+name+".png",dpi=200)
    except:
        os.makedirs(graphics_folder)
        print(" Directed %s created"%(graphics_folder))
        plt.savefig(fname=graphics_folder+name+".png",dpi=200)
    
    #plt.savefig('/home/soporte/Pictures/d2022241/sp01_f0/hola.png')
    plt.close()

path   = '/media/soporte/PROCDATA/CAMPAIGN/d2022241/sp11_f1/spec-001661749283.hdf5'
filename=path
code   = '1'
nc     = 600
nrange = 1000
f = h5py.File(path,'r')
name = filename.split("/")[-1][5:-5]
#tiempo = str(datetime.fromtimestamp(int(name)))

ch0 = np.array(f['pw0_C%s'%(code)])
ch1 = np.array(f['pw1_C%s'%(code)])

f.close()

FullSpectra_a = np.fft.fftshift(ch0,axes=0).T

#FullSpectra_b = np.fft.fftshift(ch1,axes=0).T

##MAL
#FullSpectra_a = np.fft.fftshift(ch0).T
FullSpectra_b = np.fft.fftshift(ch1).T

power_a = np.fft.ifftshift(FullSpectra_a.T,axes=0)

print("CONDICION:",power_a == ch0)

Freqs = np.linspace(-5,5,nc)
Ranges = np.linspace(0,1500,nrange)

min_a = np.min(FullSpectra_a)
min_b = np.min(FullSpectra_b)

NoiseFloor_a = np.median( FullSpectra_a, axis=0)
NoiseFloor_b = np.median( FullSpectra_b, axis=0)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_a), cmap='jet')
plt.colorbar()
plt.xlabel("Frecuencia")
plt.ylabel("Alturas")
plt.title("%s "%("CH0-ori"))

plt.subplot(1,2,2)
plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(power_a,axes=0).T), cmap='jet')
plt.colorbar()
plt.xlabel("Frecuencia")
plt.ylabel("Alturas")
plt.title("%s "%("CH0-Filt"))

plt.show()

