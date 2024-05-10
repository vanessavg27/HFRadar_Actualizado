import h5py,math
import time
import os,sys
import numpy as np 
import numpy
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

path = os.path.split(os.getcwd())[0]
sys.path.append(path)
print "REVISAR LA LINEA DE EJEMPLO COMENTADA DENTRO DE ESTE SCRIPT EN CASO TENGA PROBLEMAS DE EJECUCION"
yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
daybefore= yesterday.strftime("%Y/%m/%d")
today = datetime.datetime.now().strftime("%Y/%m/%d")

parser = argparse.ArgumentParser()
########################## PATH - MOMENTS DATA  ###################################################################################################
parser.add_argument('-path',action='store',dest='path_lectura',help='Directorio de Datos de Momentos \
	.Por defecto, se esta ingresando entre comillas /home/igp-114/Pictures/',default='/home/igp-114/Pictures/')
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
paths       = str(results.path_lectura)
freq       = results.f_freq
campaign   = results.c_campaign
code       = int(results.code_seleccionado)
dates       = results.date_seleccionado
lo         = results.lo_seleccionado

class Agent:
    def __init__(self,path,filename,channel):
        self.channel = channel
        self.filename= filename
        self.path=path
        self.LIST_AI=[]
        self.len_perfiles=self.getLongTP()
        self.getListofPeaks()

    def getDataHF(self,path,filename,attr):
        # attr for example mode 1'Data/data_SNR/channel0' , Data/data_param/channel0'
        # attr for example mode 0 'snr' 'doppler' 'noise')
        # mode 1 for HF0 y HFS 0
        dir_file= path+"/"+filename
        fp      = h5py.File(dir_file,'r')
        array = fp[attr].value
        print array
        #array    = fp.get(attr)[()]
        fp.close()
        return array

    def getLongTP(self):
        all_snr=self.getDataHF(path=self.path,filename=self.filename,attr='Data_SNR/Channel%s'%(self.channel))
        print("shape",all_snr.shape)
        return all_snr.shape[1]
        
    def getListofPeaks(self):
        self.LIST_AI=[]
        k=0
        snr =self.getDataHF(path=self.path,filename=self.filename,attr='Data_SNR/Channel%s'%(self.channel))
        j = 0
        for j in range(self.len_perfiles):
            if int(self.channel) == 0:
                peaks ,snr_peaks =find_peaks(snr[133:290,j],0.3) #SNR Station JROA
            else:
                peaks ,snr_peaks =find_peaks(snr[133:290,j],0.3) #SNR Station JROA
            if len(peaks)!=0:
                index_max = numpy.argmax(snr_peaks['peak_heights'])
                index_0   = 0
                index=[j,index_0,index_max]
                peaks = peaks+133
                self.LIST_AI.append([k,j,peaks[index_0],peaks[index_max],round(snr_peaks['peak_heights'][index_0],2),round(snr_peaks['peak_heights'][index_max],2)])
                k=k+1


class AgentHF2M:

    def __init__(self,lista_T):
        self.lista_T = lista_T      
        self.matrix_D=None
        self.matrix_D_r=None
        self.matrix_N=None
        self.matrix_N_r=None
        self.New_LD=[]
        self.New_LN=[]
        self.len_LD=0
        self.len_LN=0
        self.createListN()
        self.matrix_D,self.matrix_D_r=self.createMatrix(self.len_LD)
        self.matrix_N,self.matrix_N_r=self.createMatrix(self.len_LN)
        self.putDistanceMatrix(lista=self.New_LD,matrix=self.matrix_D)
        self.putDistanceMatrix(lista=self.New_LN,matrix=self.matrix_N)

    def createMatrix(self,n):
        matrix=[]
        i = 0
        for i in range(n):
            matrix.append([0] * n)
        matrix_r = list(map(lambda i: list(map(lambda j: j, i)),matrix))
        i = 0
        for i in range(n):
            for j in range(n):
                matrix_r[i][j]=j
        return matrix,matrix_r

    def createListN(self):
        k=0
        i = 0
        for i in range(len(self.lista_T)):
            if self.lista_T[i][1]<480: 
                if k==0:
                    temp=numpy.array([k,self.lista_T[i][0],self.lista_T[i][3],self.lista_T[i][1]])
                    self.New_LD.append(temp)
                    k=k+1
                else:
                    j=0
                    for j in range(len(self.lista_T[i][2:4])):
                        temp=numpy.array([k,self.lista_T[i][0],self.lista_T[i][2+j],self.lista_T[i][1]])
                        self.New_LD.append(temp)
                        k=k+1
        self.len_LD=len(self.New_LD)

        k=0
        i = 0
        for i in range(len(self.lista_T)):
            if 960<self.lista_T[i][1]: 
                if k==0:
                    temp=numpy.array([k,self.lista_T[i][0],self.lista_T[i][3],self.lista_T[i][1]])
                    self.New_LN.append(temp)
                    k=k+1
                else:
                    j = 0
                    for j in range(len(self.lista_T[i][2:4])):
                        temp=numpy.array([k,self.lista_T[i][0],self.lista_T[i][2+j],self.lista_T[i][1]])
                        self.New_LN.append(temp)
                        k=k+1
        self.len_LN=len(self.New_LN)

    def putDistanceMatrix(self,lista,matrix):
        n=len(lista)
        print("valor N",n)
        i=0
        j=0
        for i in range(n):
            for j in range(n):
                if i==j:
                    matrix[i][j]=0
                    if i==0 and j==0:
                        index_AI= lista[i][1]
                        if index_AI>0:
                            tmp_i=index_AI
                        else:
                            tmp_i=0
                    z=0
                if i>j:
                    matrix[i][j]=numpy.inf
                if i<j:
                    index_AI= lista[i][1]#numero de perfil
                    try:
                        len_AI  = len(self.lista_T[index_AI+1][2:4])
                    except:
                        len_AI  = -numpy.inf
                    pos     =2*(index_AI-tmp_i)+1
                    if pos<=j<pos+len_AI:
                        #print(i,j,z,index_AI)
                        if 100>(self.lista_T[index_AI][3]-self.lista_T[index_AI][2])>3.5  and (self.lista_T[index_AI][5]-self.lista_T[index_AI][4])>3.5: 
                            if z==0:
                                matrix[i][j]=numpy.inf
                            else:
                                matrix[i][j]=abs(self.lista_T[index_AI+1][2+z]-lista[i][2])
                        else:
                            matrix[i][j]=abs(self.lista_T[index_AI+1][2+z]-lista[i][2])
                        z=z+1
                    else:
                        matrix[i][j]= numpy.inf

    def getMatrixDN(self):
        return self.matrix_D,self.matrix_D_r,self.matrix_N,self.matrix_N_r

def floyd_warshall(G,G2,nV):
    distance = list(map(lambda i: list(map(lambda j: j, i)), G))
    distance2 = list(map(lambda i: list(map(lambda j: j, i)), G2))

    # Adding vertices individually
    k=0
    i=0
    j=0
    for k in range(nV):
        print(k)
        for i in range(nV):
            for j in range(nV):
                if( distance[i][k] + distance[k][j]<distance[i][j]):
                    distance[i][j]=distance[i][k] + distance[k][j]
                    distance2[i][j] = k
    return distance,distance2

def print_lindex(inicio,fin,M):
    Final=[]
    Final.append(fin)#
    while(fin!=inicio):
        if(M[inicio][fin]==fin):
            Final.append(inicio)
            fin=inicio
        else:
            fin = M[inicio][fin]
            Final.append(fin)
    Final.reverse()
    return Final

#Setting flag for frequency used to store the results of experiment in the respective folder
if freq <3:
	ngraph= 0
else:
	ngraph= 1

from datetime import datetime
day = datetime.strptime(dates, "%Y/%m/%d")
Days = day.strftime("%Y%j")

#Setting the exact folder from where the moment data will be read
location_dict = {11:"JRO_A", 12: "JRO_B", 21:"HYO_A", 22:"HYO_B", 31:"MALA",
				41:"MERCED", 51:"BARRANCA", 61:"OROYA"}
identifier = 'sp%s1_f%s'%(code, ngraph)
#Setting the exact folder where the graphics will be located
figpath	   = 'GRAPHICS_FLOYD_%s'%(location_dict[lo])
Channels = ['0','1']

for channel in Channels:
    print "iniciando con canal: ", channel
    path    = paths+identifier+'/'
    filename = str('M%s'%(Days)+'.hdf5')
    print path+filename
    if lo in [11,12,21,22]:
        if int(channel) == 0:
            lo = lo
        else:
            lo = lo
    else:
        if int(channel) == 1:
            if lo in [31, 41, 51, 61]:
                lo = lo + 1   

    directorio_SNR = "/home/igp-114/Pictures/%s/%s/d%s/"%(figpath, identifier,Days)
    try:
        os.mkdir(directorio_SNR)
    except OSError:
        print("La creacion del directorio %s fallo"%(directorio_SNR))
    else:
        print("Se ha creado el directorio: %s "%(directorio_SNR))
    
    path_fig = "/home/igp-114/Pictures/%s/%s/d%s/"%(figpath, identifier,Days)
    #path_fig = "/home/soporte/Pictures/%s/%s/d%s/"%(figpath,identifier,Days)
    
    agent = Agent(path=path,filename=filename, channel=channel)
    agent.LIST_AI[:]
   
    LISTA_AI= agent.LIST_AI
    agent2=AgentHF2M(lista_T=LISTA_AI)
    d,d_r,n,n_r=agent2.getMatrixDN()
    agent2.New_LD[:10]
    #METODO FLOYD WARSHALL       
    #MODO DIA
    d_f,d_r_f=floyd_warshall(G=d,G2=d_r,nV=len(d[0]))
    fin=len(d_r_f[0])
    print("fin",fin)
    final=print_lindex(0,fin-1,d_r_f)
    print("Longitud de la solucion",len(final))
    R=agent2.New_LD
    print("len_R",len(R))
    LIST_SOL=[]
    i = 0
    for i in range(len(final)):
        LIST_SOL.append([R[final[i]][3],R[final[i]][2]*1.5])
    #LECTURA MOMENTOS
    dir_file = path+"/"+filename
    fp       = h5py.File(dir_file,'r')
    snr      = fp.get('Data_SNR/Channel%s'%(channel))[()]
    doppler  = fp.get('Data_Doppler/Channel%s'%(channel))[()]
    noise    = fp.get('Data_Noise/Channel%s'%(channel))[()]
    power    = fp.get('Data_Pow/Channel%s'%(channel))[()]
    utime    = fp.get('utime/Channel%s'%(channel))[()]
    date_time     = numpy.arange(snr.shape[1],dtype='float')
    fp.close()

    print(snr.shape)
    print(doppler.shape)
    print(noise.shape)
    print(power.shape)
    print(date_time.shape)
    ajustar_tiempo = len(date_time)
    #CONTINUACION MODO DIA
    
    List_dia=[]
    LIST_VERIFs=[]
    i = 0
    for i in range(len(LIST_SOL)):
        LIST_VERIFs.append(LIST_SOL[i][0])

    sol_referencia = numpy.zeros(479)
    i = 0
    for i in range(479):
        if i in LIST_VERIFs:
            indice_dia = LIST_VERIFs.index(i)
            if LIST_SOL[indice_dia][1]<200:
                sol_referencia[i] = numpy.nan
            else:
                sol_referencia[i] = LIST_SOL[indice_dia][1]/1.5
        else:
            sol_referencia[i] = numpy.nan

    #print(sol_referencia)
    print(sol_referencia.shape)

    #MODO NOCHE

    n_f,n_r_f=floyd_warshall(G=n,G2=n_r,nV=len(n[0]))

    fin_n=len(n_r_f[0])
    print("fin",fin_n)
    final_n=print_lindex(0,fin_n-1,n_r_f)
    print("Longitud de la solucion",len(final_n))
    R_N=agent2.New_LN
    LIST_SOL_N=[]
    i = 0
    for i in range(len(final_n)):
        LIST_SOL_N.append([R_N[final_n[i]][3],R_N[final_n[i]][2]*1.5])
    #print(LIST_SOL_N)
    print(len(LIST_SOL_N))

    #SOLUCION TOTAL

    LIST_VERIF_TOTAL=[]
    LIST_SOL_TOTAL = LIST_SOL+LIST_SOL_N
        
    i = 0
    for i in range(len(LIST_SOL_TOTAL)):
        LIST_VERIF_TOTAL.append(LIST_SOL_TOTAL[i][0])

    sol_referencia_total = numpy.zeros(ajustar_tiempo) #AJUSTAR

    i = 0

    for i in range(ajustar_tiempo): #AJUSTAR
        if i in LIST_VERIF_TOTAL:
            indice_total = LIST_VERIF_TOTAL.index(i)
            if LIST_SOL_TOTAL[indice_total][1]<200:
                sol_referencia_total[i] = numpy.nan
            else:
                sol_referencia_total[i] = LIST_SOL_TOTAL[indice_total][1]/1.5
        else:
            sol_referencia_total[i] = numpy.nan

    #print(sol_referencia_total)
    print(sol_referencia_total.shape)

    #EJE TIEMPO
    import time, datetime
    from datetime import datetime
    j = 0
    for j in range(len(sol_referencia_total)):
        #print(j)
        date = datetime.strptime(time.ctime(int(utime[j])), '%a %b %d %H:%M:%S %Y')
        t0 = date.hour+date.minute/60.0
        date_time[j] = t0

    #CREA LISTA DE PICOS DE ACUERDO A FLOYD WARSHALL

    list_snr     = []
    list_doppler  = []
    list_noise    = []
    list_power    = []
    list_h       = []

    #OBTENIENDO DATOS DE DIFERENTES MOMENTOS

    #VELOCIDAD DOPPLER
    j = 0
    for j in range(len(sol_referencia_total)):
        if numpy.isnan(sol_referencia_total[j]):
            valor_doppler = numpy.nan
        else:
            valor_doppler = doppler[:,j][int(sol_referencia_total[j])]

        list_doppler.append(valor_doppler)

    print(list_doppler)
    print(len(list_doppler))

    #PLOT DEL DOPPLER


    #POTENCIA
    j = 0
    for j in range(len(sol_referencia_total)):
        if numpy.isnan(sol_referencia_total[j]):
            valor_power = numpy.nan
        else:
            valor_power = 10*numpy.log10(power[:,j][int(sol_referencia_total[j])])
            #10*numpy.log10(power[:,j][peaks])

        list_power.append(valor_power)

    print(list_power)
    print(len(list_power))

    #PLOT POWER


    #ALTURA
    j = 0
    for j in range(len(sol_referencia_total)):
        if numpy.isnan(sol_referencia_total[j]):
            valor_altura = numpy.nan
        else:
            valor_altura = sol_referencia_total[j]*1.5
            #list_noise_n.append(valor_n_noise)
        list_h.append(valor_altura)

    #PLOT ALTURA


    #SNR
    j = 0
    for j in range(len(sol_referencia_total)):
        if numpy.isnan(sol_referencia_total[j]):
            valor_snr = numpy.nan
        else:
            valor_snr = 10*numpy.log10(snr[:,j][int(sol_referencia_total[j])])

        list_snr.append(valor_snr)

    print(list_snr)
    print(len(list_snr))

    #PLOT SNR


    #NOISE
    j = 0
    for j in range(len(sol_referencia_total)):
        if numpy.isnan(sol_referencia_total[j]):
            valor_noise = numpy.nan
        else:
            valor_noise = 10*numpy.log10(noise[0][j])

        list_noise.append(valor_noise)

    print(list_noise)
    print(len(list_noise))
    
    
    directorio_crear = "/home/igp-114/RTDI_A/graphics_schain/%s/d%s/"%(identifier,Days)
    #directorio_crear = "/home/soporte/RTDI_A/graphics_schain/%s/d%s/"%(identifier,Days)
    try:
        os.mkdir(directorio_crear)
    except OSError:
        print("La creacion del directorio %s fallo"%(directorio_crear))
    else:
        print("Se ha creado el directorio: %s "%(directorio_crear))

    ruta_rtdi = "/home/igp-114/RTDI_A/graphics_schain/%s/d%s/"%(identifier,Days)

              
    def create(filename_out,freq,lo,channel, identifier):
        fo=open(filename_out+'.out','w+')
        fo.write('\n')
        fo.write('JICAMARCA RADIO OBSERVATORY - IGP - OUT FILE\n')
        fo.write("Station: %d ,Frequency: %2.4f, Code: %d, Channel: %d \n\n"%(int(lo),float(freq),int(identifier[2]),int(channel)))
        line = 'N     Time(Hour)      Delay(Km)     V_Doppler(m/s)      Signal_Power(dB)      Noise+Interference(dB)      SNR(dB)'
        fo.write(line+'\n\n')
        fo.close()
    def writeParameters(filename_out,N,time,delay,v_doppler,s_power,n_i,snr):
        fo=open(filename_out+'.out','a')
        line="%4d     %8.3f     %3.1f     %2.2f     %2.4f     %2.4f     %2.4f"%(N,time,delay,v_doppler,s_power,n_i,snr)
        fo.write(line+'\n')
        fo.close()

    if int(identifier[-1]) == 0:
        frequency = 272
    else: 
        frequency = 364
    ###CREA OUT###
    print "Verificar -station"
    if lo in [11,12,21,22]:
        #out- H202218611272010.out
        filename_out=ruta_rtdi+"H"+str(Days)+str(lo)+str(frequency)+str(identifier[2])+str(channel)+'0'
        create(filename_out, freq=freq,lo=lo,channel=channel, identifier = identifier)
    else:
        filename_out=ruta_rtdi+"H"+str(Days)+str(lo)+str(frequency)+str(identifier[2])+str(channel)+'0'
        create(filename_out, freq=freq,lo=lo,channel=channel, identifier=identifier)
    
    ###############
    #DATOS OUT
    altura_out      = numpy.arange(snr.shape[1],dtype='float64')
    doppler_out     = numpy.arange(snr.shape[1],dtype='float64')
    potencia_out    = numpy.arange(snr.shape[1],dtype='float64')
    snr_out         = numpy.arange(snr.shape[1],dtype='float64')
    noise_out       = numpy.arange(snr.shape[1],dtype='float64')
    j = 0
    for j in range(len(sol_referencia_total)):
        date = datetime.strptime(time.ctime(int(utime[j])), '%a %b %d %H:%M:%S %Y')
        t0 = date.hour+date.minute/60.0
        date_time[j] = t0
        if numpy.isnan(sol_referencia_total[j]):
            altura_out[j] = numpy.nan
            doppler_out[j] = numpy.nan
            potencia_out[j] = numpy.nan
            snr_out[j] = numpy.nan
            noise_out[j] = numpy.nan
        else:
            altura_out[j] = sol_referencia_total[j]*1.5
            doppler_out[j] = doppler[:,j][int(sol_referencia_total[j])]
            potencia_out[j] = 10*numpy.log10(power[:,j][int(sol_referencia_total[j])])
            snr_out[j] = 10*numpy.log10(snr[:,j][int(sol_referencia_total[j])])
            noise_out[j] = 10*numpy.log10(noise[0][j])
        print("SHOW :",j,"time :",date_time[j],"altura :",altura_out[j],"doppler :",doppler_out[j],"potencia :",potencia_out[j],"noise :",noise_out[j],"snr :",snr_out[j])
        writeParameters(filename_out=filename_out,N=j,time=date_time[j],delay=altura_out[j],v_doppler=doppler_out[j],s_power=potencia_out[j],n_i=noise_out[j],snr=snr_out[j])

    #########
    #SE VUELVEN A IMPORTAR LAS LIBRERIAS
    import datetime
    from datetime import timedelta
    import matplotlib.dates as mdates

    #PARAMETROS
    alts = np.ones((1000))
    i = 0
    for i in range(len(alts)):
        alts[i] = i*1.5
    alts = alts.tolist()
    snr=np.array(snr)
    doppler = np.array(doppler)
    print("snr",snr)
    print("doppler",doppler)

    #TIEMPO PARA RTI
    utime_rti = np.array(utime)
    date_time_objects = []
    ts = 0
    for ts in utime_rti:
        date_time_obj = datetime.datetime.fromtimestamp(ts)
        date_time_objects.append(date_time_obj)

    #PLOT RTI ALTURA VS SNR
    #fig, ax = plt.subplots(figsize=(5, 5)) #(20,10)
    fig = plt.figure(figsize=(5.4,3.2))
    fig, ax = plt.subplots()
    clrs= ax.pcolormesh(mdates.date2num(date_time_objects), alts, snr, cmap='jet')
    ax.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_format)
    ax.set_xlabel("UTC (h)", fontsize=7)
    ax.set_ylabel("Alturas (km)", fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)    
    plt.xlabel('Hora Local',fontsize=7)
    plt.ylabel('Alturas (km)',fontsize=7)
    fig_title = 'Channel%s Signal Noise Ratio (SNR) %s'%(channel,day)
    plt.title(fig_title, fontsize=7)
    # This simply sets the x-axis data to diagonal so it fits better.
    fig.autofmt_xdate()
    box=ax.get_position()
    cbarax=fig.add_axes([box.x0+box.width+0.01, box.y0, 0.025, box.height])
    cb=plt.colorbar(clrs,cax=cbarax)
    cb.set_label('SNR',fontsize=7)
    cb.mappable.set_clim(-3,15) # (-3,15) escala colores
    plt.savefig(path_fig+str(Days)+str(lo)+"777"+str(frequency)+"20001"+".jpeg")
    
    #PLOT RTI ALTURA VS DOPPLER
    #fig, ax2 = plt.subplots(figsize=(5, 5))
    fig = plt.figure(figsize=(5.4,3.2))
    fig, ax2 = plt.subplots()
    clrs2= ax2.pcolormesh(mdates.date2num(date_time_objects), alts, doppler, cmap='jet')
    ax2.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M')
    ax2.xaxis.set_major_formatter(date_format)
    ax2.set_xlabel("UTC (h)", fontsize=7)
    ax2.set_ylabel("Alturas (km)", fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig_title = 'Channel%s : %s'%(channel,day)
    plt.title(fig_title, fontsize=7)
    # This simply sets the x-axis data to diagonal so it fits better.
    fig.autofmt_xdate()
    box=ax2.get_position()
    cbarax2=fig.add_axes([box.x0+box.width+0.01, box.y0, 0.025, box.height])
    cb2=plt.colorbar(clrs2,cax=cbarax2)
    cb2.mappable.set_clim(-60,60) #-60,60
    cb2.set_label('V Doppler', fontsize=7)
    plt.savefig(path_fig+str(Days)+str(lo)+"555"+str(frequency)+"20001"+".jpeg")

    import datetime, time, math
    from datetime import timedelta
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    from matplotlib.pyplot import imshow
    #LECTURA MOMENTOS
    def bubbleSort(alist):
        for passnum in range(len(alist)-1,0,-1):
            for i in range(passnum):
                if alist[i]>alist[i+1]:
                    temp=alist[i]
                    alist[i]=alist[i+1]
                    alist[i+1]=temp

    dir_file = path+"/"+filename
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
    profile = range(1000)
    rango   = range(1000)
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
        print "Perfiles %s - %s -Ch%s"%(i,utime_minute[i],channel)
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
        layer1=tmp[m/2] # this has a value from sorting 7 values.
        icount1=(icount1+1)%m
        data_img_genaro.append(rango[layer1])# how to know that is value is also int? from 0 to 1000?

    data_time_genaro = [ ((int(y.second)/60.0 + y.minute)/60.0 + y.hour  +0.000278) for y in utime_minute]#User1 was here
    print "data_img_genaro", len(data_img_genaro)
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
    plt.title("RTDI %s"%(utime_minute[-1]))
    plt.xlabel("Local Time (hr)",color="k")
    plt.ylabel("Range (km)",color="k")

    plt.yticks(range(ymin,ymax+1,100))
    im=plt.imshow(spc_db[int(ymin/1.5):int(ymax/1.5)-1, xmin*60: xmax*60-1,:].astype(numpy.uint64),origin='lower',aspect='auto',extent=[xmin, xmax, ymin,ymax])#')
    plt.savefig(ruta_rtdi+str(Days)+str(lo)+"5"+str(frequency)+str(identifier[2])+str(channel)+".jpeg")

    del date
