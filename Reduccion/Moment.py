#!/usr/bin/env python3
"""
Este codigo permite convetir espectros a momentos empleando filtro en alturas para
reducir variabilidad en potencia, ademas se agrego funcionalidad para generar
los archivos hdf5 automticamente uno despues de otro, tambien se corrigio el removerDC
"""
#Forma de ejecucion para espectros normales
#python3 Moment.py  -path "/media/soporte/DATA1/" -f 2.72216796875 -lo 41 -C 1 -code 0 -date "2022/08/30" -R 0 -graphics_folder "/media/soporte/PROCDATA/MERCED/MomentsFloyd/"
#Forma de ejecucion para espectros reducidos
#python3 Moment.py  -path "/media/soporte/DATA1/" -f 2.72216796875 -lo 41 -C 1 -code 0 -date "2022/08/30" -R 1 -graphics_folder "/media/soporte/PROCDATA/MERCED/MomentsFloyd/"

import h5py
import math,time,os,sys,numpy
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import datetime
import time
from datetime import datetime, timedelta
from scipy.signal import find_peaks_cwt,medfilt,find_peaks
from scipy.sparse import csr_matrix
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
												21: HYO-N45O, 22: HYO-N45E', default=12)
########################## GRAPHICS - RESULTS  ###################################################################################################
parser.add_argument('-graphics_folder',action='store',dest='graphics_folder',help='Directorio de Resultados \
					.Por defecto, se esta ingresando entre comillas /media/soporte/PROCDATA/MomentsFloyd/', default="/media/soporte/PROCDATA/MomentsFloyd/")

parser.add_argument('-R',action='store',dest='reduccion',type=int,help='Parametro para obtener el valor del ruido SNR de los espectros filtrados \
		            siendo 1 para espectros filtrados y 0 para espectros originales.', default=0)

#Parsing the options of the script
results	   = parser.parse_args()
path	   = str(results.path_lectura) #/media/igp-114/PROCDATA/
freqs	   = results.f_freq            # 2.72216796875
campaign   = results.c_campaign        # Mode 0(normal) y 1(Campaign)
code	   = int(results.code_seleccionado)
Days	   = results.date_seleccionado
lo		   = results.lo_seleccionado
graphics_folder = results.graphics_folder
R          = results.reduccion

if R == 1:
    graphics_folder=graphics_folder[:-1]+"_Filt/"
    print("Final Path: ",graphics_folder)
 
if campaign == 1:
    path= path+"CAMPAIGN/"
    nc  = 600
else:
    path = path
    nc = 100

if freqs <3:
    ngraph = 0
else:
    ngraph = 1

from datetime import datetime
days = datetime.strptime(Days,"%Y/%m/%d")
day = days.strftime("%Y%j")
print(day)

###########################################################

###########################################################

def getDatavaluefromDirFilename(path,file,value):
    dir_file= path+"/"+file
    fp      = h5py.File(dir_file,'r')
    array    = fp.get(value)[()]
    fp.close()
    return array

def read_newdata(path,filename,nc,value):
    dir_file = path+'.hdf5'
    fp       = h5py.File(dir_file,'r')
    dir_file = filename+'/'+value
    #print("Filename:",filename)

    if value[:2] == 'pw':        
        matrix_read    = np.array(fp[dir_file])
        power          = matrix_read[:,2] #Value power
        f2             = matrix_read[:,1] #axis heights
        f1             = matrix_read[:,0] #axis profiles

        array = csr_matrix((power,(f2,f1)),shape= (nc,1000))
        array = array.toarray()
        array[array == 0] = read_newdata(path,filename,nc,'noise'+str(chan)+'_C'+str(code))
    else:
        array = fp.get(dir_file)[()]
    fp.close()
    return array

def plot(snr,peaks,tiempo):

    plt.plot(snr, label = 'original')
    #plt.plot(peaks_newindex,)
    plt.plot(peaks, snr[peaks], 'o', mfc= 'none', label = 'max')
    #plt.plot(peaks, snr[peaks], 'o', mfc= 'none', label = 'ori')
    plt.title(tiempo)
    plt.show()

def peaks_V2(snr,high_index):
    prom  = np.mean(snr)
    snr_2 = snr - prom
    snr_2 = np.where(snr_2<=0.0, 0, snr_2)
    index_peak  = [index for index,value in enumerate(snr_2) if value > prom]

    for i in range(len(index_peak)-1):
        for j in range(len(index_peak)-1):
            if snr[index_peak[j+1]]>snr[index_peak[j]]:
                val             = index_peak[j]
                index_peak[j]   = index_peak[j+1]
                index_peak[j+1] = val
    #print("B:",b)
    index_peak = [ n for n in index_peak if (n > high_index and n < 801)]
    return index_peak

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

def GetRGBData(data_spc, threshv=0.083):
    #This method if called overwrites the data image read from the HDF5
    #s
    s = np.fft.fftshift(data_spc,axes=1)
    s = s.transpose()
    L= s.shape[0] # Number of profiles
    N= s.shape[1] # Number of heights
    data_RGB = numpy.zeros([3,N])
    im=int(math.floor(L/2)) #50
    i0l=im - int(math.floor(L*threshv)) #10
    i0h=im + int(math.floor(L*threshv)) #90
#
#    Freqs = np.linspace(-5,5,L)
#    Ranges = np.linspace(0,1500,N)
#    plt.figure(figsize=(10,6))
#    plt.pcolormesh(Ranges, Freqs, np.log10(s), cmap='jet')
    
#    plt.colorbar()
#    plt.xlabel("Frecuencia")
#    plt.ylabel("Alturas")
#    plt.show()


    for ri in numpy.arange(N):
        data_RGB[0, ri]= numpy.sum(s[0:i0l,ri])
        data_RGB[1, ri]= numpy.sum(s[i0l:i0h,ri])
        data_RGB[2, ri]= numpy.sum(s[i0h:L,ri])
    
    return data_RGB

#Para datos filtrados
def GetRGBData_filt(data_spc, threshv=0.167):
    s = np.fft.fftshift(data_spc,axes=1)
    s = s.transpose()
    #s = data_spc.transpose()
    L= s.shape[0] # Number of profiles
    N= s.shape[1] # Number of heights
#
#    Freqs = np.linspace(-5,5,L)
#    Ranges = np.linspace(0,1500,N)
#    plt.figure(figsize=(10,6))
#    plt.pcolormesh(Ranges, Freqs, np.log10(s), cmap='jet')
    
#    plt.colorbar()
#    plt.xlabel("Frecuencia")
#    plt.ylabel("Alturas")
#    plt.show()
#
    data_RGB = numpy.zeros([3,N])
    im=int(math.floor(L/2)) #Normal_mode:50 Campaign_mode:300
    low_index = im - int(math.floor(L*0.25))
    up_index  = im + int(math.floor(L*0.25))

    lon = up_index-low_index
    i0l = im - int(math.floor(lon*threshv))
    i0h = im + int(math.floor(lon*threshv))
    for ri in numpy.arange(N):
        data_RGB[0, ri]= numpy.sum(s[low_index:i0l,ri])
        data_RGB[1, ri]= numpy.sum(s[i0l:i0h,ri])
        data_RGB[2, ri]= numpy.sum(s[i0h:up_index,ri])
    
    return data_RGB

def GetImageSNR(data_input):
    image    = data_input.transpose()
    noise    = list(range(3))
    snr      = list(range(3))
    sn1      = -20.0  #-10.0
    sn2      = 40.0
    npy      = image.shape[1]
    nSamples = 1000.0
    r2       = min(nSamples,image.shape[0]-1)
    #print("r2:",r2)
    r1       =int(r2*0.9)
    #print("r1:",r1)
    ncount=0.0
    noise[0]=noise[1]=noise[2]=0.0
    for i in range(r1,r2):
        ncount += 1
        for j in range(npy):
            noise[j]+=image[i,j]
    #print("ncount",ncount)

    for j in range(npy):
        noise[j]/=ncount
    #print("Noise RGB 10% ",noise)
    buffer2=numpy.zeros((1000,3),dtype='float')
    for i in range(r2+1):
        for j in range(npy):
            snr[j]=(image[i,j]-noise[j])/noise[j]
            if (snr[j]> 0.01):
                #print("SNR-RGB:",10.0*math.log10(snr[j]))
                snr[j]=(10.0*math.log10(snr[j])-sn1)/(sn2-sn1)
            else:
                snr[j]=0.0
        buffer2[i]=snr[:]
    data_img_snr = buffer2
    return data_img_snr
    

#def data(folder,directorio,nc,Days,code):
folder = "sp%s1_f%s"%(code,ngraph)
path     = path+"d"+day+'/'+folder

print(path)
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

if R == 1:
    try:
        f        = h5py.File(path+'.hdf5','r')
        filelist = sorted(list(f.keys()))
        f.close()
    except OSError:
        print("    NO EXISTE EL ARCHIVO DE LECTURA:",path+'.hdf5')
        exit()
else:
    filelist   = sorted(os.listdir(path))

ntime      = len(filelist)
utime      = numpy.empty(shape=(ntime))
out_dir = graphics_folder+"%s/"%(folder)
try:
    os.makedirs(out_dir)
except:
    print("    Carpeta existente.",end="-*-")
#out_dir = graphics_folder+"%s/"%(folder)
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
i = 0
from datetime import datetime

for chan in Channels:
    j=0
    for filename in filelist:
        nsets=int(nc/nc) #600
        k=0
        npeaks=5
        
        if R ==1:
            four  = read_newdata(path,filename,nc,value='pw'+str(chan)+'_C'+str(code)).swapaxes(0,1)
            utime[j] = read_newdata(path,filename,nc,value='t')
            noise = read_newdata(path,filename,nc,'noise'+str(chan)+'_C'+str(code))
            new_pspec = pspec = four
            name = filename[5:]

        else:
            four  = getDatavaluefromDirFilename(path=path,file=filename,value='pw'+str(chan)+'_C'+str(code)).swapaxes(0,1)
            utime[j]= getDatavaluefromDirFilename(path=path,file=filename,value='t')
            new_pspec = pspec = four
            
            #Metodo de hallar el ruido
            prom_pspec= numpy.mean(new_pspec,1) #cambio a new_pspec
            noise = numpy.median(prom_pspec[0:200])
            name = filename[5:-5]

        
        l = 0

        date = datetime.strptime(time.ctime(int(utime[j])),'%a %b %d %H:%M:%S %Y')
        t0 = date.hour+date.minute/60.0
        
        #import datetime
        tiempo= str(datetime.fromtimestamp(int(name)))

        

        # ---------- CALCULO DEL RGB desde el pspec -------------------#

        #data_RGB     = GetRGBData(new_pspec, threshv=0.083)
        data_RGB      = GetRGBData_filt(new_pspec, threshv=0.083)
        #print("*** DATA-RGB",data_RGB)
        data_IMG_SNR = GetImageSNR(data_input= data_RGB).transpose()
        #data_IMG_SNR = GetRGBData_filt(data_input= data_RGB).transpose()

        #-----------------------------------------------------
        i = 0

        for i in range(nrange):
            #print("aNTES DE RUIDO",new_pspec)
            new_pspec[i,:] = new_pspec[i,:]-noise             #cambio a new_pspec
            #print("DESPUES DE RUIDO",new_pspec)
            pow_signal[i]= numpy.sum(new_pspec[i,:])/nc        #cambio a new_pspec
            power        = numpy.sum(new_pspec[i,:])/nc        # AQUI DEBO DIVIDIR ENtre NC=600, cambio a new_pspec
            #print freq
            doppler[i] = numpy.sum(vmax*freq*new_pspec[i,:])/(power*nc)    # 600 cambio a new_pspec
            snr[i] = (power)/(noise)

        # find peaks
        #peaks = find_peaks_cwt(snr, numpy.arange(20,60))
        peaks = peaks_V2(snr,50)
        #print("PEAKS",peaks, peaks[0])
        if ( int(name) > 1661893300):
            pass
            #plot(snr,peaks,tiempo)
            #print(peaks)
        if len(peaks)<6:
            #npeaks = len(peaks)
            peaks=numpy.array(peaks)
            if len(peaks)==4:
                npeaks = 5
                peaks= numpy.append(peaks,peaks[3])
            if len(peaks) == 3:
                pico = numpy.copy(peaks)
                peaks= numpy.append(peaks,pico[1])
                peaks= numpy.append(peaks,pico[2])
                npeaks = len(peaks)
                del pico
            if len(peaks) == 2:
                pico = numpy.copy(peaks)
                peaks= numpy.append(peaks,pico[0])
                peaks= numpy.append(peaks,pico[1])
                peaks= numpy.append(peaks,pico[1])
                npeaks = len(peaks)
                del pico

            if len(peaks) == 1:
                for i in range(npeaks-1):
                    peaks= numpy.append(peaks,peaks[0])
                npeaks = len(peaks)

        else:
            peaks = numpy.array(peaks[:npeaks]) # skip ground peak
        #print(len(peaks),peaks)
        if len(peaks) == 0:
            peaks    =  numpy.empty((5))
            peaks[:] =  numpy.nan
            print(filename,folder,chan,j,"No hay datos espectrales SNR = Nan")
        else:
            layer = peaks[0]
            print(filename,folder,chan,j,layer,layer*1.5,doppler[layer])
        j=j+1
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
        
        #if i == 0:
        #    peaks_log=peaks.reshape(npeaks,1)
        #    i += 1
        #else:
        #    peaks_log=numpy.concatenate((peaks_log,peaks.reshape(npeaks,1)),axis=1)


        try:
            #print("PEAKS:",npeaks)
            #print(peaks_log)
            peaks_log=numpy.concatenate((peaks_log,peaks.reshape(npeaks,1)),axis=1)
        except NameError:
            peaks_log=peaks.reshape(npeaks,1)


        try:
            #print("Forma data image",data_RGB.reshape(1,3,nrange))
            RGB = numpy.concatenate((RGB,data_IMG_SNR.reshape(1,3,nrange)),axis=0)
            #RGB = numpy.concatenate((RGB,data_RGB.reshape(1,3,nrange)),axis=0)
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
   
 
