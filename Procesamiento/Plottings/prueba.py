#!/usr/bin/env python3.6

import h5py
import math,time,os,sys,numpy
import numpy as np 
import argparse
from datetime import datetime, timedelta
from scipy.signal import find_peaks_cwt
from scipy.signal import find_peaks_cwt,medfilt,find_peaks
from scipy.signal import savgol_filter
import datetime
import pandas as pd
from datetime import datetime

filepath = "/media/soporte/PROCDATA/CAMPAIGN/d2022195/sp21_f1/"
day="d2022195"
filename = "spec-001657803877.hdf5"
file = filepath + filename
campaign = 1
code = 2
freqs = 3.64
if campaign == 1:
    path = "/media/soporte/PROCDATA/CAMPAIGN/"
    nc = 600
else:
    path = "/media/soporte/PROCDATA/"
    nc = 100

if freqs < 3:
    ngraph = 0
else:
    ngraph = 1

ch = "0"


def getDatavaluefromDirFilename(file,value):
    #dir_file= path+"/"+file
    dir_file = file
    fp      = h5py.File(dir_file,'r')
    array    = fp.get(value)[()]
    fp.close()
    return array

def GetRGBData(data_spc, threshv=1.5):
    s=data_spc.transpose()
    L=s.shape[0]#Number of profiles
    N=s.shape[1]#Number of heights
    data_RGB = numpy.zeros([3,N])
    
    
    pass



folder = "sp%s1_f%s"%(code,ngraph)
path = path+day+"/"+folder
print(path)
nrange = 1000
lpf = 0.08
#--------------------------
pspec = numpy.zeros((nrange,nc))
new_pspec = numpy.zeros((nrange,nc))
doppler    = numpy.empty(shape=(nrange))
snr        = numpy.empty(shape=(nrange))
pow_signal = numpy.empty(shape=(nrange))
filelist   = os.listdir(path)
ntime      = len(filelist)
utime      = numpy.empty(shape=(ntime))

#--------------------------
freq = float("%se6"%(freqs))
vmax=0.5*1.0e1*3.0e8/freq # 10 Hz sampling, assumes backscatter
npeaks=5

freq = numpy.fft.fftfreq(nc)

#print("freq-fft",freq)

channels = ['0','1']

for chan in channels:
    j = 0
    filename = '/media/soporte/PROCDATA/CAMPAIGN/d2022195/sp21_f1/spec-001657803877.hdf5'
    nsets = int(nc/nc)
    k = 0
    print('pw'+str(chan)+'_C'+str(code))
    for k in range(nsets):
        print("K",k, "nsets: ",nsets)
        four = getDatavaluefromDirFilename(file=filename,value='pw'+str(chan)+'_C'+str(code)).swapaxes(0,1)
        
        pspec += four[:,k*nc:(k+1)*nc]
        

    print("pspec: ",pspec)
    
    for l in range(nrange):
        if l == 0:
            new_pspec[0]= pspec[0]
        elif(l<nrange -1):
            new_pspec[l] = pspec[l-1]+pspec[l]
        else:
            new_pspec[l]=pspec[l]

    utime[j]= getDatavaluefromDirFilename(file=filename,value='t')
    date = datetime.strptime(time.ctime(int(utime[j])),'%a %b %d %H:%M:%S %Y')
    t0 = date.hour+date.minute/60.0

    j=j+1
    prom_pspec= numpy.mean(new_pspec,1) #cambio a new_pspec
    noise = numpy.median(prom_pspec[0:200])
    #print (prom_pspec.shape)
    # ---------- CALCULO DEL RGB desde el pspec -------------------#
    data_RGB     = GetRGBData(new_pspec, threshv=1.5)
    print("DATA_RGB",data_RGB)
    data_IMG_SNR = GetImageSNR(data_input= data_RGB).transpose()
    print("data_IMG_SNR",data_IMG_SNR)
        #-----------------------------------------------------
    i = 0
    for i in range(nrange):
        new_pspec[i,:] = new_pspec[i,:]-noise              #cambio a new_pspec
        pow_signal[i]= numpy.sum(new_pspec[i,:])/nc        #cambio a new_pspec
        power = numpy.sum(new_pspec[i,:])/nc               # AQUI DEBO DIVIDIR ENtre NC=600, cambio a new_pspec
        #print freq
        doppler[i] = numpy.sum(vmax*freq*new_pspec[i,:])/(power*100)    # 600 cambio a new_pspec
        snr[i] = power/(noise)

        # find peaks
        peaks = find_peaks_cwt(snr, numpy.arange(20,60))
        print("PEAKS",peaks, peaks[0])