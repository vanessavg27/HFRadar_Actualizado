#!/usr/bin/env python
"""
Este codigo permite convetir espectros a momentos empleando filtro en alturas para
reducir variabilidad en potencia, ademas se agrego funcionalidad para generar
los archivos hdf5 automticamente uno despues de otro, tambien se corrigio el removerDC
"""

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

print ("*** Inicinado PROC - MOMENTS ***")
path = os.path.split(os.getcwd())[0]
sys.path.append(path)

yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
daybefore = yesterday.strftime("%Y/%m/%d")

parser = argparse.ArgumentParser()
########################## PATH- DATA  ###################################################################################################
parser.add_argument('-path',action='store',dest='path_lectura',help='Directorio de Datos \
					.Por defecto, se esta ingresando entre comillas /media/soporte/PROCDATA/',default='/media/soporte/PROCDATA/')
########################## FRECUENCIA ####################################################################################################
parser.add_argument('-f',action='store',dest='f_freq',type=float,help='Frecuencia en Mhz 2.72 y 3.64. Por defecto, se esta ingresando 2.72 ',default=2.72216796875)
########################## CAMPAIGN ### 600 o 100 perfiles ###############################################################################
parser.add_argument('-C',action='store',dest='c_campaign',type=int,help='Campaign 1 (600 perfiles) y 0(100 perfiles). Por defecto, se esta ingresando 1',default=1)
########################## CODIGO - INPUT ################################################################################################
parser.add_argument('-code',action='store',dest='code_seleccionado',type=int,help='Code de Tx para generar en estacion \
										de Rx Spectro 0,1,2. Por defecto, se esta ingresando 0(Ancon)',default=0)
########################## DAY- SELECCION ################################################################################################
parser.add_argument('-date',action='store',dest='date_seleccionado',help='Seleccionar fecha si es OFFLINE se ingresa \
							la fecha con el dia deseado. Por defecto, considera el dia anterior',default=daybefore)
########################## LOCATION AND ORIENTATION ####################################################################################################
parser.add_argument('-lo',action='store',dest='lo_seleccionado',type=int,help='Parametro para establecer la ubicacion de la estacion de Rx y su orientacion.\
										Example: XA   ----- X: Es el primer valor determina la ubicacion de la estacion. A: Es \
										  el segundo valor determina la orientacion N45O o N45E.  \
										11: JRO-N450, 12: JRO-N45E \
												21: HYO-N45O, 22: HYO-N45E', default=11)
########################## GRAPHICS - RESULTS  ###################################################################################################
parser.add_argument('-graphics_folder',action='store',dest='graphics_folder',help='Directorio de Resultados \
					.Por defecto, se esta ingresando entre comillas /home/soporte/Pictures/', default='/home/soporte/Pictures/')

#Parsing the options of the script
results	   = parser.parse_args()
path	   = str(results.path_lectura) #/media/igp-114/PROCDATA/
freqs	   = results.f_freq            # 2.72216796875
campaign   = results.c_campaign        # Mode 0(normal) y 1(Campaign)
code	   = int(results.code_seleccionado)
Days	   = results.date_seleccionado
lo		   = results.lo_seleccionado
graphics_folder = results.graphics_folder

if campaign == 1:
    path = "/media/soporte/PROCDATA/CAMPAIGN/"
    nc = 600
else:
    path = "/media/soporte/PROCDATA/"
    nc = 100

if freqs <3:
    ngraph = 0
else:
    ngraph = 1

from datetime import datetime
days = datetime.strptime(Days, "%Y/%m/%d")
day = days.strftime("%Y%j")
print(day)

################################################################

################################################################

def getDatavaluefromDirFilename(path,file,value):
    dir_file= path+"/"+file
    fp      = h5py.File(dir_file,'r')
    array    = fp.get(value)[()]
    fp.close()
    return array

def whiten_spec(S):
    S=S.swapaxes(0,1)
    n_rep=S.shape[0]
    # whiten spectrum
    for i in range(n_rep):
        m0=numpy.median(S[i,:])
        s0=numpy.median(numpy.abs(S[i,:]-m0))
        S[i,:]=(S[i,:]-m0)/s0
    S=S.swapaxes(0,1)
    return(S)

def GetRGBData(data_spc, threshv=1.5):
    #This method if called overwrites the data image read from the HDF5
    s=data_spc.transpose()
    L= s.shape[0] # Number of profiles
    N= s.shape[1] # Number of heights
    data_RGB = numpy.zeros([3,N])
    im=int(math.floor(L/2)) #50
    i0l=im - int(math.floor(L*threshv)) #10
    i0h=im + int(math.floor(L*threshv)) #90

    for ri in numpy.arange(N):
        data_RGB[0, ri]= numpy.sum(s[0:i0l,ri])
        data_RGB[1, ri]= numpy.sum(s[i0l:i0h,ri])
        data_RGB[2, ri]= numpy.sum(s[i0h:L,ri])
    return data_RGB

def GetImageSNR(data_input):
    image    = data_input.transpose()
    noise    = list(range(3))
    snr      = list(range(3))
    sn1      = -10.0  #-10.0
    sn2      = 20.0
    npy      = image.shape[1]
    nSamples = 1000.0
    r2       = min(nSamples,image.shape[0]-1)
    r1       =int(r2*0.9)
    ncount=0.0
    noise[0]=noise[1]=noise[2]=0.0
    for i in range(r1,r2):
        ncount += 1
        for j in range(npy):
            noise[j]+=image[i,j]

    for j in range(npy):
        noise[j]/=ncount
    buffer2=numpy.zeros((1000,3),dtype='float')
    for i in range(r2):
        for j in range(npy):
            snr[j]=(image[i,j]-noise[j])/noise[j]
            if (snr[j]> 0.01):
                snr[j]=(10.0*math.log10(snr[j])-sn1)/(sn2-sn1)
            else:
                snr[j]=0.0
        buffer2[i]=snr[:]
    data_img_snr = buffer2
    #print("DATA-IMG-SNR: ",data_img_snr)
    return data_img_snr
    #"Aqui anadiendo lo puntos blancos"

#def data(folder,directorio,nc,Days,code):
folder = "sp%s1_f%s"%(code,ngraph)
path     = path+"d"+day+'/'+folder
print (path)
#------------------------------------------------------------------------------------------------------
nrange=1000
lpf = 0.08        # just look at lowest frequency bins w/i [+/- 0.5] range
code = code
# Doppler considerations
freq = float("%se6"%(freqs))
#print(freq)
vmax=0.5*1.0e1*3.0e8/freq # 10 Hz sampling, assumes backscatter
npeaks=5
freq = numpy.fft.fftfreq(nc)
#----------------------------------------------------------------------------------------------------------
data_RGB     = numpy.zeros([3,nrange])
data_IMG_SNR = numpy.zeros([3,nrange])

#----------------------------------------
pspec = numpy.zeros((nrange,nc))
new_pspec = numpy.zeros((nrange,nc))
doppler    = numpy.empty(shape=(nrange))
snr        = numpy.empty(shape=(nrange))
pow_signal = numpy.empty(shape=(nrange))
filelist   = os.listdir(path)
ntime      = len(filelist)
utime      = numpy.empty(shape=(ntime))
out_dir = "/media/soporte/PROCDATA/MomentsFloyd/%s/"%(folder)
print(out_dir)
print(day)
outfile    = out_dir+'M'+str(day)+'.hdf5'
print("outfile",outfile)
#---------------------------------------------------------------------------------------------
informacion = "Tx%s-f%s - Station: %s - Fecha: %s"%(code,ngraph,lo,day)
hf = h5py.File(outfile, 'w')
hf.create_dataset('Metadata', data=informacion)
POWER = hf.create_group("Data_Pow")
rgb = hf.create_group("Data_RGB")
NOISE = hf.create_group("Data_Noise")
PEAKS = hf.create_group("Data_Peaks")
SNR = hf.create_group("Data_SNR")
DOPPLER = hf.create_group("Data_Doppler")
UTIME = hf.create_group("utime")
Channels = ['0','1']

for chan in Channels:
    j=0
    for filename in sorted(os.listdir(path)):
        nsets=int(nc/nc) #600
        k=0
        print(filename)
        for k in range(nsets):
            four= getDatavaluefromDirFilename(path=path,file=filename,value='pw'+str(chan)+'_C'+str(code)).swapaxes(0,1)
            pspec +=four[:,k*nc:(k+1)*nc]
        l = 0
        #print("PSPEC: ",pspec)
        
        for l in range(nrange):
            if l==0:
                new_pspec[0]= pspec[0]
            elif(l<nrange-1):
                new_pspec[l]=pspec[l-1]+pspec[l]+pspec[l+1]
            else:
                new_pspec[l]=pspec[l]
        from datetime import datetime
        utime[j]= getDatavaluefromDirFilename(path=path,file=filename,value='t')
        date = datetime.strptime(time.ctime(int(utime[j])),'%a %b %d %H:%M:%S %Y')
        t0 = date.hour+date.minute/60.0
        
        j=j+1
        prom_pspec= numpy.mean(new_pspec,1) #cambio a new_pspec
        noise = numpy.median(prom_pspec[0:200])
        #print (prom_pspec.shape)
        # ---------- CALCULO DEL RGB desde el pspec -------------------#
        data_RGB     = GetRGBData(new_pspec, threshv=1.5)
        data_IMG_SNR = GetImageSNR(data_input= data_RGB).transpose()
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
        #print("PEAKS",peaks, peaks[0])
        #print(peaks)
        if len(peaks)<6:
            peaks=numpy.array(peaks)
            if len(peaks)==4:
                peaks= numpy.append(peaks,peaks[3])
        else:
            peaks = numpy.array(peaks[1:npeaks+1]) # skip ground peak
        #print(len(peaks),peaks)
        #print("Peak-after: ",peaks, peaks[0], sep="***")
        layer = peaks[0]
        print(filename,folder,chan,j,layer,layer*1.5,doppler[layer])

        # export
        try:
            power_out = numpy.concatenate((power_out,pow_signal.reshape(nrange,1)),axis=1)
        except NameError:
            power_out = pow_signal.reshape(nrange,1)

        try:
            noise_out = numpy.concatenate((noise_out,noise.reshape(1,1)),axis=1)
        except NameError:
            noise_out = noise.reshape(1,1)

        try:
            snr_out = numpy.concatenate((snr_out,snr.reshape(nrange,1)),axis=1)
        except NameError:
            snr_out = snr.reshape(nrange,1)

        try:
            doppler_out = numpy.concatenate((doppler_out,doppler.reshape(nrange,1)),axis=1)
        except NameError:
            doppler_out = doppler.reshape(nrange,1)

        try:
            peaks_log=numpy.concatenate((peaks_log,peaks.reshape(npeaks,1)),axis=1)
        except NameError:
            peaks_log=peaks.reshape(npeaks,1)

        try:
            RGB = numpy.concatenate((RGB,data_IMG_SNR.reshape(1,3,nrange)),axis=0)
        except NameError:
            RGB = data_IMG_SNR.reshape(1,3,nrange)
            # zero out power spectrum
        pspec = numpy.zeros((nrange,nc))
        new_pspec = numpy.zeros((nrange,nc))
    hf = h5py.File(outfile, 'r')            
    des = POWER.create_dataset('Channel%s'%(chan),data=power_out,maxshape=(None,None),dtype='float32')
    del power_out
    des1 = rgb.create_dataset('Channel%s'%(chan),data=RGB,maxshape=(None,None,None),dtype='float32')
    del RGB
    des2 = NOISE.create_dataset('Channel%s'%(chan),data=noise_out,maxshape=(None,None),dtype='float32')
    del noise_out
    des3 = PEAKS.create_dataset('Channel%s'%(chan),data=peaks_log,maxshape=(None,None),dtype='int32')
    del peaks_log
    des4 = SNR.create_dataset('Channel%s'%(chan),data=snr_out,maxshape=(None,None),dtype='float32')
    del snr_out
    des5 = DOPPLER.create_dataset('Channel%s'%(chan),data=doppler_out,maxshape=(None,None),dtype='float32')
    del doppler_out
    des6 = UTIME.create_dataset('Channel%s'%(chan), data=utime,maxshape=(None,),dtype='float' )       
    date = datetime.strptime(time.ctime(int(utime[0])),'%a %b %d %H:%M:%S %Y')
    t0 = date.hour+date.minute/60.0
    date = datetime.strptime(time.ctime(int(utime[-1])),'%a %b %d %H:%M:%S %Y')
    t1 = date.hour+date.minute/60.0
    print("**** Guardando Momentos *****", out_dir) 
    hf.close()
hf.close()
   
 