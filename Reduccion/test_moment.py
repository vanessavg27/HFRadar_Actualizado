#!/usr/bin/env python3
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

def getDatavaluefromDirFilename(path,file,value):
    dir_file= path+"/"+file
    fp      = h5py.File(dir_file,'r')
    array    = fp.get(value)[()]
    print("FUNCTION",array.shape)
    fp.close()
    return array

def GetRGBData(data_spc, threshv=1):
    #This method if called overwrites the data image read from the HDF5
    s=data_spc.transpose()
    L= s.shape[0] # Number of profiles
    N= s.shape[1] # Number of heights
    data_RGB = np.zeros([3,N])
    im=int(math.floor(L/2)) #50
    i0l=im - int(math.floor(L*threshv)) #10
    i0h=im + int(math.floor(L*threshv)) #90
    print(s.shape)
    print("i0l",i0l)
    print("i0h",i0h)
    print(s[0:i0l,0])
    #for ri in np.arange(N):

    
        
        #data_RGB[0, ri]= np.sum(s[0:i0l,ri])
        #data_RGB[1, ri]= np.sum(s[i0l:i0h,ri])
        #data_RGB[2, ri]= np.sum(s[i0h:L,ri])
    
    print
    #return data_RGB

#path = "/media/soporte/PROCDATA/MomentsFloyd_Filt/sp21_f0/"

path = "/media/soporte/PROCDATA/CAMPAIGN/d2022241/sp01_f0"
nc = 600
nrange= 1000
channels = ['0','1']
filename = "spec-001661749283.hdf5"
#filename = "spec-00"
new_pspec = np.zeros((nrange,nc))
print()
code= '0'
for chan in channels:
    new_pspec = np.zeros((nrange,nc))
    nset = int(nc/nc)
    k = 0
    npeaks = 5
    #four = getDatavaluefromDirFilename(path=path, file=filename,value='pw'+str(chan)+'_C'+str(code)).swapaxes(0,1)
    four = getDatavaluefromDirFilename(path=path, file=filename,value='pw'+str(chan)+'_C'+str(code))
    print(four.shape)
    print("-*-")
    pspec = four.T
    Freqs = np.linspace(-5,5,nc)
    Ranges = np.linspace(0,1500,nrange)
    print(pspec.shape)
    #print("-*-")
    #print(pspec[0])
    l = 0
    for l in range(nrange):
        if l==0:
            new_pspec[0]= pspec[0]
        elif(l<nrange-1):
            new_pspec[l]=(pspec[l-1]+pspec[l]+pspec[l+1])/3 #Probando la division/3
        else:
            new_pspec[l]=pspec[l]
    #j=j+1
    prom_pspec= np.mean(new_pspec,1)
    #noise = np.median(prom[0:200])
    

    ########################
    ########## RGB ##########
    #sa = np.fft.fftshift(new_pspec)
    sa = new_pspec.transpose()

    correct = np.fft.fftshift(new_pspec,axes=1).transpose()
    print("FORMA-RGB",sa.shape)

    
    #pspec = pspec.T
    
    plt.figure(figsize=(20,10))
    plt.subplot(1,4,1)
    new_pspec = new_pspec.T
    pspec = pspec.T
    print("PSPEC.shape",pspec.shape)
    print("NEW-PSPEC.shape",new_pspec.shape)
    plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pspec,axes=0).T), cmap='jet')
    #plt.pcolormesh(Freqs, Ranges, np.log10(pspec), cmap='jet')
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.title("%s "%("Power"))
    
    plt.subplot(1,4,2)
    plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(new_pspec,axes=0).T), cmap='jet')
    #plt.pcolormesh(Freqs, Ranges, np.log10(new_pspec.transpose()), cmap='jet')
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.title("%s "%("Regulacion de ruido"))
    #plt.show()

    plt.subplot(1,4,3)
    plt.pcolormesh(Ranges, Freqs, np.log10(sa), cmap='jet')
    #plt.pcolormesh(Freqs, Ranges, np.log10(new_pspec.transpose()), cmap='jet')
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.title("%s "%("Regulacion de ruido"))

    plt.subplot(1,4,4)
    plt.pcolormesh(Ranges, Freqs, np.log10(correct), cmap='jet')
    #plt.pcolormesh(Freqs, Ranges, np.log10(new_pspec.transpose()), cmap='jet')
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.show()


    del four
    del pspec
    del new_pspec

    

    #GetRGBData(new_pspec, threshv=1)
    #print("DATA-RGB",data_RGB)



