import datetime, time, math, os
from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.pyplot import imshow
import h5py,math
import time, os, sys, numpy
import numpy as np 
import argparse
from datetime import datetime, timedelta
from scipy.signal import find_peaks_cwt
from scipy.signal import find_peaks_cwt,medfilt,find_peaks
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import datetime
import pandas as pd

path = os.path.split(os.getcwd())[0]
sys.path.append(path)
print("REVISAR LA LINEA DE EJEMPLO COMENTADA DENTRO DE ESTE SCRIPT EN CASO TENGA PROBLEMAS DE EJECUCION")
yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
daybefore= yesterday.strftime("%Y/%m/%d")
today = datetime.datetime.now().strftime("%Y/%m/%d")

parser = argparse.ArgumentParser()
########################## PATH - MOMENTS DATA  ###################################################################################################
parser.add_argument('-path',action='store',dest='path_lectura',help='Directorio de Datos de Momentos \
	.Por defecto, se esta ingresando entre comillas /media/soporte/PROCDATA/',default='/media/igp-114/PROCDATA/MomentsFloyd/')
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
	21: HYO-N45O, 22: HYO-N45E',default=11)
###########################################################################################################################################

#Parsing the options of the script
results    = parser.parse_args()
path       = str(results.path_lectura)
freq       = results.f_freq
campaign   = results.c_campaign
code       = int(results.code_seleccionado)
dates       = results.date_seleccionado
lo         = results.lo_seleccionado

#Setting flag for frequency used to store the results of experiment in the respective folder
if freq <3:
    ngraph= 0
    frequency = 272
else:
    ngraph= 1
    frequency = 364

if lo %2 == 0:
    station_type = 'B'
else:
    station_type = 'A'

from datetime import datetime
day = datetime.strptime(dates, "%Y/%m/%d")
Days = day.strftime("%Y%j")

#folder
identifier = 'sp%s1_f%s'%(code, ngraph)

directorio_crear = "/home/soporte/RTDI_%s/graphics_schain/%s/d%s/"%(station_type,identifier,Days)
#os.makedirs(directorio_crear)
try:
    #os.mkdir(directorio_crear)
    os.makedirs(directorio_crear)
except FileExistsError:
    print("Carpeta existente: %s "%(directorio_crear))
else:
    print("Se ha creado el directorio: %s "%(directorio_crear))

ruta_rtdi = "/home/soporte/RTDI_%s/graphics_schain/%s/d%s/"%(station_type,identifier,Days)
#LECTURA MOMENTOS
def bubbleSort(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                temp=alist[i]
                alist[i]=alist[i+1]
                alist[i+1]=temp
                

import datetime, time, math
from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.pyplot import imshow

filename = str('M%s'%(Days)+'.hdf5')
Channels = ['0','1']
for channel in Channels:
    if lo in [11,12,21,22]:
        if int(channel) == 0:
            lo = lo
        else:
            lo = lo
    else:
        if int(channel) == 1:
            if lo in [31, 41, 51, 61]:
                lo = lo + 1   
    dir_file = path+identifier+'/'+filename
    print("dir_file",dir_file)
    fp       = h5py.File(dir_file,'r')
    RGB      = fp.get('Data_RGB/Channel%s'%(channel))[()]	
    utime    = fp.get('utime/Channel%s'%(channel))[()]
    fp.close()
    print(RGB.shape) #1438,3,1000
    # 1000,1440,3 spc_db
    #spc_db = numpy.empty((RGB.shape(2),RGB.shape(0),RGB.shape(1))
    spc_db = numpy.empty((1000,1440,3))
    utime_minute     =  [datetime.datetime.fromtimestamp(x) for x in utime ]
    minute = [int(x.hour*60 + x.minute + x.second/60.0) for x in utime_minute]
    xmin = 0
    xmax = 24
    ymin = 0
    ymax = 1500
    for i in range(len(minute)):
        for j in range(1000): #de 0 a 1000
            ir=max(min(int(RGB[i][0][j]*255),255),0)
            ig=max(min(int(RGB[i][1][j]*255),255),0)
            ib=max(min(int(RGB[i][2][j]*255),255),0)
            if (ir+ig+ib)>0: # aqui podemos insertar un filtro de color
                spc_db[j,minute[i],:] =(ir,ig,ib)

    data_img_genaro = []
    NRANGE  = 1000
    layer1=None
    layer2=None
    profile = list(range(1000))
    rango   = list(range(1000))
    dr      = 1.5
    for i in range(NRANGE):
        rango[i]=float(i*dr)
    queue1=[0,0,0,0,0,32655,32655]
    tmp=[0,0,0,0,0,0,0]
    icount1=0
    max_deriv=0
    i = 0
    j= 0
    k = 0
    for i in range(len(minute)):
        print("Perfiles %s - %s -Ch%s"%(i,utime_minute[i],channel))
        for j in range(1000):
            profile[j]= RGB[i][0][j]+RGB[i][1][j] + RGB[i][2][j]
        #if i <= 480 or i >=960:
        max_deriv=0
        for k in range(1000):
            if (rango[k]>=200.0 and rango[k]<450.0):
                deriv=(-2.0*profile[k-2]-profile[k-1]+profile[k+1]+2.0*profile[k+2])/10.0 #10.04
                if (deriv>max_deriv):
                    max_deriv=deriv
                    layer1=k
        queue1[icount1]=layer1
        m=7
        for l in range(7):
            tmp[l]=queue1[l]
        bubbleSort(tmp)
        layer1=tmp[int(m/2)] # this has a value from sorting 7 values.
        icount1=(icount1+1)%m
        data_img_genaro.append(rango[layer1])# how to know that is value is also int? from 0 to 1000?

    data_time_genaro = [ ((int(y.second)/60.0 + y.minute)/60.0 + y.hour  +0.000278) for y in utime_minute]#User1 was here
    print("data_img_genaro", len(data_img_genaro))
    i=0
    for i in range(len(data_time_genaro)):# Could be data from F Region or and Region too
        time_minutes=int(data_time_genaro[i]*60)#escalar
        rango_layer=int(data_img_genaro[i]/1.5)#escalar
        xticksSperation = 120 # was 120hardcoded
        #Position where the maximum reflection happens
        height_high = min(999,rango_layer+2)
        height_low =  max(0,rango_layer-2)
        time_high = min(time_minutes+2, 1439)
        time_low = max(0, time_minutes-2)
        spc_db[height_low:height_high,time_low:time_high,:] =(811,811,811)
    print("ploteo")

    plt.clf()
    lim1 = max(1, xmin*60)
    lim2 = min(spc_db.shape[1]-1,xmax*60-2 )
    aux = numpy.zeros((int(ymax/1.5)-1-int(ymin/1.5),xmax*60-1 -  xmin*60, 3))
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = [6.4,4.8]
    #print(utime_minute)
    plt.title("RTDI %s"%(utime_minute[-1]))
    plt.xlabel("Local Time (hr)",color="k")
    plt.ylabel("Range (km)",color="k")

    plt.yticks(range(ymin,ymax+1,100))
    im=plt.imshow(spc_db[int(ymin/1.5):int(ymax/1.5)-1, xmin*60: xmax*60-1,:].astype(numpy.uint64),origin='lower',aspect='auto',extent=[xmin, xmax, ymin,ymax])#')
    plt.savefig(ruta_rtdi+str(Days)+str(lo)+"5"+str(frequency)+str(identifier[2])+str(channel)+".jpeg")
    del data_img_genaro
