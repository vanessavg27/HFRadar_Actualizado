import matplotlib as mpl
mpl.rc('text', usetex = False)
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import h5py
import scipy.constants as const
from scipy.signal import medfilt, find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev

from time import time
import datetime
import os

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
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

import datetime

def read_newdata(path,filename,nc,value):
    dir_file = path+'.hdf5'
    fp       = h5py.File(dir_file,'r')
    dir_file = filename+'/'+value
    print("Filename:",filename)

    if value[:2] == 'pw':        
        matrix_read    = np.array(fp[dir_file])
        power          = matrix_read[:,2] #Value power
        f2             = matrix_read[:,1] #axis profiles
        f1             = matrix_read[:,0] #axis heights

        array = csr_matrix((power,(f2,f1)),shape= (nc,1000))
        array = array.toarray()
        array[array == 0] = read_newdata(path,filename,nc,'min'+str(chan)+'_C'+str(code))
        array = array.swapaxes(0,1)
    else:
        array = fp.get(dir_file)[()]
    fp.close()
    return array

PathToData = '/media/soporte/PROCDATA/Same_Struct/CAMPAIGN/d2022241/sp21_f0/spec-001661749283.hdf5'
code='2'
nc=600
nrange=1000
f = h5py.File(PathToData,'r')
pw_ch0 = np.array(f['pw1_C%s'%(code)])
print("FORMA:",pw_ch0.shape)
FullSpectra_a = np.fft.fftshift(pw_ch0,axes=0).T
print("Forma fft y trans:",FullSpectra_a.shape)

Freqs = np.linspace(-5,5,nc)
Ranges = np.linspace(0,1500,nrange)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch0,axes=0).T), cmap='jet')
plt.colorbar()
plt.xlabel("Frecuencia")
plt.ylabel("Alturas")
plt.title("%s "%("CH0-ori"))

plt.subplot(1,2,2)
plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_a), cmap='jet')
plt.colorbar()
plt.xlabel("Frecuencia")
plt.ylabel("Alturas")
plt.title("%s "%("CH0-ori"))
#plt.show()

#path = '/media/soporte/PROCDATA/CAMPAIGN/d2022241/Same_Struct/sp01_f0/'
def getDatavaluefromDirFilename(path,file,value):
    dir_file= path+"/"+file
    fp      = h5py.File(dir_file,'r')
    array    = fp.get(value)[()]
    fp.close()
    return array

ruta_dir = '/media/soporte/PROCDATA/CAMPAIGN/d2022241/sp21_f0.hdf5'
path= '/media/soporte/PROCDATA/CAMPAIGN/d2022241'
folder = 'sp21_f0'

path = path+'/'+folder

#f = h5py.File(ruta_dir,'r')
#f = h5py.File(path+'.hdf5','r')
channel=['0','1']
code = '2'
nrange = 1000
nc = 600
Freqs = np.linspace(-5,5,nc)
Ranges = np.linspace(0,1500,1000)

#fileslist = sorted(list(f.keys()))
filelist = sorted(list(h5py.File(path+'.hdf5','r').keys()))
h5py.File(path+'.hdf5','r').close()
ntime    = len(filelist)
utime    = np.empty(shape=(ntime))
print("*****************************")
#print("F.KEYS:",files,len(f.keys()))
for chan in channel:
    j = 0
    for filename in filelist:
        print("J:",j," - ",filename)
        #value = 'pw'+str(chan)+'_C'+str(code)
        #matrix_read = np.array(f[filename+'/pw'+chan+'_C'+code])

        four     = read_newdata(path,filename,nc,value='pw'+str(chan)+'_C'+str(code))
        utime[j] = read_newdata(path,filename,nc,value='t')
        print("UTIME:",utime[j])
        #power = matrix_read[:,2]
        #f2    = matrix_read[:,1]
        #f1    = matrix_read[:,0]

        #matrix_read = csr_matrix((power,(f2,f1)),shape= (1000,nc))
        #matrix_read = matrix_read.toarray()
        #matrix_read[matrix_read == 0] = f[filename+'/min'+chan+'_C'+code]

        plt.figure(figsize=(6,6))
        plt.pcolormesh(Freqs, Ranges, np.log10((four)),cmap='jet')
        plt.colorbar()
        plt.show()
        j+=1


    

spec = 'spec-001661749283'
#######
chan = '0'
matrix_read = np.array(f[filename+'/pw'+chan+'_C'+code])
power = matrix_read[:,2]
f2    = matrix_read[:,1]
f1    = matrix_read[:,0]




ruta_dir = '/media/soporte/PROCDATA/d2022238/sp01_f0.hdf5'
f = h5py.File(ruta_dir,'r')
channel=['0','1']
code = '0'
nrange= 1000
nc = 100
Freqs = np.linspace(-5,5,nc)
Ranges = np.linspace(0,1500,1000)
spec = 'spec-001661576286'
#######
chan = '0'
matrix_read = np.array(f[spec+'/pw'+chan+'_C'+code])
power = matrix_read[:,2]
f2    = matrix_read[:,1]
f1    = matrix_read[:,0]
matrix_build = csr_matrix((power,(f2,f1)),shape = (nc,nrange))
#matrix_build = csr_matrix((power,(f2,f1)),shape = (nrange,nc))
matrix_build = matrix_build.toarray()
matrix_build[matrix_build == 0] = f[spec+'/min'+chan+'_C'+code]
print("MATRIX BUILD FORM:",matrix_build.shape)
Spectra = np.fft.fftshift(matrix_build).T

#matrix_build = matrix_build.toarray()
#matrix_build[matrix_build == 0] = f[spec+'/min'+chan+'_C'+code]

plt.figure(figsize=(6,6))
plt.pcolormesh(Freqs, Ranges, np.log10((Spectra)),cmap='jet')

#plt.show()

#######
'''
for chan in channel:
    print("CANAL:",chan)
    for i in f.keys():
        matrix_read = np.array(f[i+'/pw'+chan+'_C'+code])
        power = matrix_read[:,2]
        f2    = matrix_read[:,1]
        f1    = matrix_read[:,0]
        matrix_build = csr_matrix((power,(f2,f1)),shape = (nrange,nc))
        matrix_build = matrix_build.toarray()
        matrix_build[matrix_build == 0] = f[i+'/min'+chan+'_C'+code]
        
        plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(matrix_build).T),cmap='jet')
        plt.colorbar()
        plt.show()
        #power_b = 
        print(i)
        #print(power_a)
'''

'''
dir_file = "/media/soporte/PROCDATA/MomentsFloyd/sp21_f0/M2022241.hdf5"
Channels = ['0','1']
for channel in Channels:
    fp = h5py.File(dir_file,'r')
    utime = fp.get('utime/Channel%s'%(channel))[()]
    fp.close()
    utime_minute = [datetime.datetime.fromtimestamp(x) for x in utime]
    minute = [int(x.hour*60 + x.minute + x.second/60) for x in utime_minute]
print("UTIME: ", utime)
print("FECHA: ", utime_minute)
'''
''' 
    FS_filtrado_A = New_Full_Spectra_a.toarray()
    FS_filtrado_A[FS_filtrado_A == 0] = min_a


    FS_filtrado_B = New_Full_Spectra_b.toarray()
    FS_filtrado_B[FS_filtrado_B == 0] = min_b

    pw_ch0 = np.fft.ifftshift(FS_filtrado_A.T)
    pw_ch1 = np.fft.ifftshift(FS_filtrado_B.T)

''' 


'''
    if clean == 1:
        FS_filtrado_A = New_Full_Spectra_a.toarray()
        FS_filtrado_A[FS_filtrado_A == 0] = min_a
        pw_ch0 = np.fft.ifftshift(FS_filtrado_A.T)
    
        FS_filtrado_B = New_Full_Spectra_b.toarray()
        FS_filtrado_B[FS_filtrado_B == 0] = min_b
        pw_ch1 = np.fft.ifftshift(FS_filtrado_B.T)

        Guardado_same(CurrentSpec, pw_ch0,pw_ch1,Noise_a,Noise_b)
''' 


















