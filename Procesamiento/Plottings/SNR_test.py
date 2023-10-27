import h5py,math
import time
import os,sys
import numpy as np
import argparse
from datetime import datetime, timedelta
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt,medfilt,find_peaks
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd
from scipy.signal import find_peaks as fp
from peakutils import indexes
from datetime import datetime
import h5py

path = os.path.split(os.getcwd())[0]
sys.path.append(path)
#print("REVISAR LA LINEA DE EJEMPLO COMENTADA DENTRO DE ESTE SCRIPT EN CASO TENGA PROBLEMAS DE EJECUCION")
#python3 SNR_test.py -path /media/soporte/PROCDATA/Reducted/JROA/MomentsFloyd_Filt/ -f 2.72216796875 -C 0 -code 0 -lo 11 -date "2023/09/17"

yesterday = datetime.now() - timedelta(days=1)
daybefore= yesterday.strftime("%Y/%m/%d")
today = datetime.now().strftime("%Y/%m/%d")

parser = argparse.ArgumentParser()
########################## PATH - MOMENTS DATA  ###################################################################################################
parser.add_argument('-path',action='store',dest='path_lectura',help='Directorio de Datos de Momentos \
	.Por defecto, se esta ingresando entre comillas /home/soporte/Pictures/',default='/media/soporte/PROCDATA/MomentsFloyd_Filt/')
########################## PATH OUT - MOMENTS DATA  ###################################################################################################
parser.add_argument('-path_o',action='store',dest='path_o',help='Directorio de Datos de Momentos \
	.Por defecto, se esta ingresando entre comillas /home/soporte/',default='/home/soporte/')
########################## FRECUENCIA ####################################################################################################
parser.add_argument('-f',action='store',dest='f_freq',type=float,help='Frecuencia en Mhz 2.72 y 3.64. Por defecto, se esta ingresando 2.72 ',default=2.72216796875)
########################## CAMPAIGN ### 600 o 100 perfiles ###############################################################################
parser.add_argument('-C',action='store',dest='c_campaign',type=int,help='Campaign 1 (600 perfiles) y 0(100 perfiles). Por defecto, se esta ingresando 1',default=0)
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
paths      = str(results.path_lectura)
path_o     = str(results.path_o)
freq       = results.f_freq
campaign   = results.c_campaign
code       = int(results.code_seleccionado)
dates       = results.date_seleccionado
lo         = results.lo_seleccionado
channel    =['0','1']

class Agent:
    def __init__(self,path,path_o,dates,channel,code,freqs,lo):
        self.channel = channel
        self.code = code
        self.Days = datetime.strptime(dates, "%Y/%m/%d").strftime("%Y%j")
        self.filename= str('M%s'%(self.Days)+'.hdf5')
        self.path=path
        self.path_o = path_o
        self.lo = lo
        self.LIST_AI=[]
        self.freq = freqs
        #Prueba se comento
        self.dates=dates
        self.Channels = channel
        #print(self.LIST_AI[300:600])
        if self.freq <3:
            self.ngraph= 0
            self.frequency = 272
        else:
            self.ngraph= 1
            self.frequency = 364

        self.identifier = str('sp%s1_f%s'%(self.code, self.ngraph))

    def ploteo(self,data):
        import datetime
        empty_peak =[]
        for i in range(self.len_perfiles):
            a=data[:,i]
            peaks1= self.GetPeaks(a,50,mdist=2,threshold=0.2)
            dt = datetime.datetime.fromtimestamp(self.tiempo[i])
            minuto = dt.hour*60+dt.minute
            print("Perfil",i,"tiempo",minuto)
            print("Picos1:",peaks1)
            #plt.plot(a)
            #plt.plot(peaks1,a[peaks1],'o', mfc= 'none')
            if len(peaks1) == 0:
                empty_peak.append(minuto)
            del(peaks1)
            del(a)
            #plt.savefig(fname="/home/soporte/Pictures/GRAPHICS_FLOYD_JRO_A/%s.png"%(minuto),dpi=250)
            #plt.close()
        print("Tiempo vacio:",empty_peak)
    
    def orden(self,data,indexs,high_down):
        for i in range(len(indexs)-1): #Ordenamiento segun el mayor pico al menor pico
            for j in range(len(indexs)-1):
                if data[indexs[j+1]]>data[indexs[j]]:
                    val             = indexs[j]
                    indexs[j]       = indexs[j+1]
                    indexs[j+1]     = val
        peaks = [ n for n in indexs if (n > high_down and n < 600)]
        return peaks

    def GetPeaks(self,data,high_down,mdist, threshold):
        prom = np.mean(data)
        snr  = data - prom
        snr  = np.where(snr<=0.0,0,snr)
        peaks1   = indexes(snr[:601],min_dist = mdist,thres=threshold)
        peaks = self.orden(snr,peaks1,high_down)
        
        if len(peaks) == 0:
            peaks = indexes(data,min_dist = mdist,thres=0.16)
            #peaks = [ n for n in peaks if (n > high_down and n < 701)]
            peaks = self.orden(snr,peaks,high_down)
        return peaks

    def getDataHF(self,attr):
        # attr for example mode 1'Data/data_SNR/channel0' , Data/data_param/channel0'
        # attr for example mode 0 'snr' 'doppler' 'noise')
        # mode 1 for HF0 y HFS 0
        dir_file= self.path+self.identifier+'/'+self.filename
        fp      = h5py.File(dir_file,'r')
        array = fp.get(attr)[()]
        fp.close()
        return array

    def getLongTP(self):
        
        all_snr=self.getDataHF(attr='Data_SNR/Channel%s'%(self.chan))
        print("shape",all_snr.shape)
        return all_snr.shape[1]
  
    def getListofPeaks2(self):
        import datetime
        self.LIST_AI=[]
        #snr    = self.getDataHF(path=self.path,filename=self.filename,attr='Data_SNR/Channel%s'%(self.channel))
        tiempo = self.getDataHF(attr='utime/Channel%s'%(self.chan))
        #print("Tiempo:",tiempo)
        j = 0
        for j in range(self.len_perfiles):
        
            dt = datetime.datetime.fromtimestamp(tiempo[j])
            minuto = dt.hour*60+dt.minute

            snr_i = self.snr[:,j]

            if int(self.chan) == 0 or int(self.chan) == 1:
                
                peaks = self.GetPeaks(snr_i,40,mdist=2,threshold=0.2) #Alturas masyores a 40*1.5Km= 60 Km
                
                ''' Usar en caso se requiera plotear los picos segun el minuto > '960'    
                    #plt.plot(snr_prueba, label = 'original')
                    #plt.plot(peaks2, snr_prueba[peaks2],'o', mfc= 'none', label = 'max')
                    #plt.show()
                '''
            else:
                print("No hay datos de SNR en Canal: Ch0 y Ch1")
            
            info = [minuto,peaks]
            self.LIST_AI.append(info)
            
    def analize(self):
        cont=-1
        N=len(self.LIST_AI)
        minimo = 999
        picos_oficiales = []
        for dato in self.LIST_AI:
            cont =cont+1
            if cont == 0:
                peak = dato[1][0]    
            elif cont==(N-1):
                peak = dato[1][0] 
            else:
                list_dist = []
                for p in dato[1]:
                    
                    dist = p - peak
                    dist1= self.LIST_AI[cont+1][1][0]-p
                    #print("Contador",cont)
                    #print("Minuto",dato[0],"pico",p)
                    #print(self.snr[:,cont][p],0.65*self.snr[:,cont][dato[1][0]])
                    #list_dist.append(abs(dist)) 
                    if len(dato[1]) >6:
                        list_dist.append(abs(dist))
                    else:
                        if self.snr[:,cont][p] > 0.55*self.snr[:,cont][dato[1][0]]:
                            list_dist.append(abs(dist))
                        else:
                            continue
                
                minimo = min(list_dist)
                index = list_dist.index(minimo)  
                peak = dato[1][index]
            
            picos_oficiales.append([cont,dato[0],peak])
        return picos_oficiales
    
    def set(self,dates,code,ngraph):

        from datetime import datetime
        day = datetime.strptime(dates, "%Y/%m/%d")
        self.Days = day.strftime("%Y%j")

        #Setting the exact folder from where the moment data will be read
        location_dict = {11:"JRO_A", 12: "JRO_B", 21:"HYO_A", 22:"HYO_B", 31:"MALA",
				         41:"MERCED", 51:"BARRANCA", 61:"OROYA"}
        identifier = 'sp%s1_f%s'%(code, ngraph)
        #Setting the exact folder where the graphics will be located
        figpath	   = 'GRAPHICS_FLOYD_%s'%(location_dict[lo])
        self.Channels = ['0','1']
        

    def run(self):

        for channel in self.Channels:
            print("")
            print("Iniciando con canal: ", channel)
            self.chan = channel
            path    = paths+self.identifier+'/'
            filename = str('M%s'%(self.Days)+'.hdf5')

            self.len_perfiles=self.getLongTP()
            self.snr=self.getDataHF(attr='Data_SNR/Channel%s'%(self.chan))
            self.getListofPeaks2()
            
            if self.lo in [11,12,21,22]:
                if self.lo % 2 == 0:
                    self.station_type='B'
                else:
                    self.station_type = 'A'
            else:
                if int(channel) == 1:
                    if self.lo in [31, 41, 51, 61]:
                        self.lo = self.lo + 1
                        self.station_type = 'B'
                    else:
                        self.station_type = 'A'
            
            datos_general = []

            self.dir_file = self.path+self.identifier+'/'+self.filename

            print("Archivo fuente:",self.dir_file)
            fp       = h5py.File(self.dir_file,'r')
            snr      = fp.get('Data_SNR/Channel%s'%(channel))[()]
            doppler  = fp.get('Data_Doppler/Channel%s'%(channel))[()]
            noise    = fp.get('Data_Noise/Channel%s'%(channel))[()]
            power    = fp.get('Data_Pow/Channel%s'%(channel))[()]
            utime    = fp.get('utime/Channel%s'%(channel))[()]
            date_time     = np.arange(snr.shape[1],dtype='float')
            fp.close()

            list_snr     = []
            list_doppler = []
            list_noise   = []
            list_power   = []
            list_h       = []

            perfiles      =   self.len_perfiles
            picos_oficiales = self.analize()
            
            for p in picos_oficiales:
        
                if p[1] <= 480:
                    datos_general.append(p)
                elif p[1] >= 960:
                    datos_general.append(p)
                else:
                    datos_general.append([p[0],p[1],np.nan])
            
            for j in range(len(datos_general)):

                if np.isnan(datos_general[j][2]):
                    valor_doppler = np.nan
                    valor_power = np.nan
                    valor_altura = np.nan
                    valor_snr = np.nan
                    valor_noise = np.nan
                else:
                    #VELOCIDAD DOPPLER
                    valor_doppler = doppler[:,j][int(datos_general[j][2])]
                    #POTENCIA
                    valor_power   = 10*np.log10(power[:,j][int(datos_general[j][2])])
                    #HIGH
                    valor_altura  = datos_general[j][2]*1.5
                    #SNR
                    valor_snr     = 10*np.log10(snr[:,j][int(datos_general[j][2])])
                    #NOISE
                    valor_noise   = 10*np.log10(noise[0][j])

                list_doppler.append(valor_doppler)
                list_power.append(valor_power)
                list_h.append(valor_altura)
                list_snr.append(valor_snr)
                list_noise.append(valor_noise)

            if int(self.identifier[-1]) == 0:
                frequency = 272
            else: 
                frequency = 364
            ###CREA OUT###
            self.ruta_rtdi = "%sRTDI_%s/graphics_schain/%s/d%s/"%(self.path_o,self.station_type,self.identifier,self.Days)
            
            try:
            #os.mkdir(directorio_crear)
                os.makedirs(self.ruta_rtdi)
            except FileExistsError:
                print("Carpeta existente: %s "%(self.ruta_rtdi))
            else:
                print("Se ha creado el directorio: %s "%(self.ruta_rtdi))

            
            if self.lo in [11,12,21,22]:
                #out- H202218611272010.out
                filename_out=self.ruta_rtdi+"H"+str(self.Days)+str(lo)+str(frequency)+str(self.identifier[2])+str(channel)+'0'
                self.create(filename_out, freq=freq,lo=lo,channel=channel, identifier = self.identifier)
            else:
                filename_out=self.ruta_rtdi+"H"+str(self.Days)+str(lo)+str(frequency)+str(self.identifier[2])+str(channel)+'0'
                self.create(filename_out, freq=freq,lo=lo,channel=channel, identifier=self.identifier)

            ###############
            #DATOS OUT
            altura_out      = np.arange(snr.shape[1],dtype='float64')
            doppler_out     = np.arange(snr.shape[1],dtype='float64')
            potencia_out    = np.arange(snr.shape[1],dtype='float64')
            snr_out         = np.arange(snr.shape[1],dtype='float64')
            noise_out       = np.arange(snr.shape[1],dtype='float64')
            j = 0
            for j in range(len(datos_general)):
                date = datetime.strptime(time.ctime(int(utime[j])), '%a %b %d %H:%M:%S %Y')
                t0 = date.hour+date.minute/60.0
                date_time[j] = t0
                if np.isnan(datos_general[j][2]):
                    #print(True,j)
                    altura_out[j] = np.nan
                    doppler_out[j] = np.nan
                    potencia_out[j] = np.nan
                    snr_out[j] = np.nan
                    noise_out[j] = np.nan
                else:
                    ''' 
                    altura_out[j] = datos_general[j][2]*1.5
                    doppler_out[j] = doppler[:,j][int(datos_general[j][2])]
                    potencia_out[j] = 10*np.log10(power[:,j][int(datos_general[j][2])])
                    snr_out[j] = 10*np.log10(snr[:,j][int(datos_general[j][2])])
                    noise_out[j] = 10*np.log10(noise[0][j])
                    '''
                    altura_out[j]   = list_h[j]
                    doppler_out[j]  = list_doppler[j]
                    potencia_out[j] = (list_power[j])
                    snr_out[j]      = (list_snr[j])
                    noise_out[j]    = (list_noise[j])            
        
                #print("SHOW :",j,"time :",date_time[j],"altura :",altura_out[j],"doppler :",doppler_out[j],"potencia :",potencia_out[j],"noise :",noise_out[j],"snr :",snr_out[j])
                self.writeParameters(filename_out=filename_out,N=j,time=date_time[j],delay=altura_out[j],v_doppler=doppler_out[j],s_power=potencia_out[j],n_i=noise_out[j],snr=snr_out[j])
            #plt.plot(date_time,altura_out)
            #plt.ylim(0,1500)
            #plt.show()
            del list_snr
            del list_doppler
            del list_noise  
            del list_power  
            del list_h
            print("Guardado en:",filename_out)  
            
    def create(self,filename_out,freq,lo,channel, identifier):
        fo=open(filename_out+'.out','w+')
        fo.write('\n\n')
        fo.write('JICAMARCA RADIO OBSERVATORY - IGP - OUT FILE\n')
        fo.write("Station: %d ,Frequency: %2.4f, Code: %d, Channel: %d \n\n"%(int(lo),float(freq),int(identifier[2]),int(channel)))
        line = 'N  Time(Hour)   Delay(Km)  V_Doppler(m/s)   Signal_Power(dB)   Noise+Interference(dB)      SNR(dB)'
        fo.write(line+'\n\n')
        fo.close()
    def writeParameters(self,filename_out,N,time,delay,v_doppler,s_power,n_i,snr):
        fo=open(filename_out+'.out','a')
        line="%4d         %8.3f         %3.1f         %2.2f         %2.4f        %2.4f       %2.4f"%(N,time,delay,v_doppler,s_power,n_i,snr)
        fo.write(line+'\n')
        fo.close()
    
    def bubbleSort(self,alist):
        for passnum in range(len(alist)-1,0,-1):
            for i in range(passnum):
                if alist[i]>alist[i+1]:
                    temp=alist[i]
                    alist[i]=alist[i+1]
                    alist[i+1]=temp

    def RTI(self):
        
        for channel in self.Channels:
            self.chan = channel
            NRANGE = 1000
            data_img_genaro = []
            RGB   = self.getDataHF(attr='Data_RGB/Channel%s'%(self.chan))
            utime = self.getDataHF('utime/Channel%s'%(self.chan))
            utime_minute = [datetime.fromtimestamp(x) for x in utime]
            minute = [int(x.hour*60 + x.minute + x.second/60.0) for x in utime_minute]
            spc_db = np.empty((1000,1440,3))

            xmin = 0;  xmax = 24;   ymin = 0;   ymax = 1500
            min_value = np.min(RGB);max_value = np.max(RGB)

            for i in range(len(minute)):
                for j in range(NRANGE): #de 0 a 1000
                    ir=max(min(int(RGB[i][0][j]*255),255),0)
                    ig=max(min(int(RGB[i][1][j]*255),255),0)
                    ib=max(min(int(RGB[i][2][j]*255),255),0)
                    if (ir+ig+ib)>0: # aqui podemos insertar un filtro de color
                        spc_db[j,minute[i],:] =(ir,ig,ib)
            
            queue1=[0,0,0,0,0,32655,32655]
            tmp   =[0,0,0,0,0,0,0]
            print(len(minute))
            icount1=0
            self.getListofPeaks2()
            picos_oficiales = self.analize()

            for i in range(len(minute)):
                print("Perfiles %s - %s -Ch%s"%(i,utime_minute[i],channel))
                layer1 = picos_oficiales[i][2]
                queue1[icount1]=layer1
                m=7
                #print("QUEUE:",queue1)
                for l in range(7):
                    tmp[l]=queue1[l]
                self.bubbleSort(tmp)
                layer1=tmp[int(m/2)]
                icount1=(icount1+1)%m
                data_img_genaro.append(picos_oficiales[i][2]) #

            data_time_genaro = [ ((int(y.second)/60.0 + y.minute)/60.0 + y.hour +0.000278) for y in utime_minute]
            
            i = 0
            for i in range(len(data_time_genaro)):# Could be data from F Region or and Region too
                time_minutes=int(data_time_genaro[i]*60)#escalar
                rango_layer=int(data_img_genaro[i])#escalar
                xticksSperation = 120 # was 120hardcoded
                #Position where the maximum reflection happens
                height_high = min(999,rango_layer+2)
                height_low =  max(0,rango_layer-2)
                time_high = min(time_minutes+2, 1439)
                time_low = max(0, time_minutes-2)
                spc_db[height_low:height_high,time_low:time_high,:] =(811,811,811)
        
            plt.clf()
            
            plt.rcParams['figure.dpi'] = 100
            plt.rcParams['figure.figsize'] = [6.4,4.8]
            #print(utime_minute)
            plt.title("RTDI %s"%(utime_minute[-1]))
            plt.xlabel("Local Time (hr)",color="k")
            plt.ylabel("Range (km)",color="k")


            im=plt.imshow(spc_db[int(ymin/1.5):int(ymax/1.5)-1, xmin*60: xmax*60-1,:].astype(np.uint64),origin='lower',aspect='auto',extent=[xmin, xmax, ymin,ymax])
            #im=plt.imshow(spc_db[int(ymin):(int(ymax)-1), xmin*60: xmax*60-1,:].astype(np.uint64),origin='lower',aspect='auto')       
            plt.yticks(range(ymin,ymax+1,100))
            plt.savefig(self.ruta_rtdi+str(self.Days)+str(lo)+"5"+str(self.frequency)+str(self.identifier[2])+str(channel)+".jpeg")

            del data_img_genaro


obj=Agent(path=paths,path_o=path_o,dates=dates, channel=channel,code=code,freqs=freq,lo=lo)
obj.run()
obj.RTI()
