#!/usr/bin/env python3
import matplotlib as mpl
mpl.rc('text', usetex = False)
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.constants as const
from scipy.signal import medfilt, find_peaks
#from scipy.ndimage import median_filter as med_filt
#from scipy.signal import savgol_filter as savgol
#from scipy.optimize import curve_fit
#from scipy.interpolate import splrep, splev
from time import time
#from datetime import datetime
import datetime
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
#from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from scipy.sparse import csr_matrix
import argparse
import glob
import math,time,os,sys

#Almacenar Ploteo 
#./Filtrado.py -f 2.72 -code 0 -C 1 -date "2022/08/29" -P 1

#Almacenar matriz de potencias reducidas (data sparse)
#./Filtrado.py -f 2.72 -code 2 -C 1 -date "2022/08/29" -R 1
#./Filtrado.py -f 2.72 -code 2 -C 1 -date "2022/08/29" -R 1 -path '/media/soporte/RAWDATA/HYOa/'


#Almacenar matriz de potencias filtrado (manteniendo la estructura)
#./Filtrado.py -f 2.72 -code 2 -C 1 -date "2022/08/29" -clean 1
#./Filtrado.py -f 2.72 -code 2 -C 1 -date "2022/08/29" -clean 1 -path_o '/media/soporte/PROCDATA/Same_Struct/'

def Ruido(input,nc,nrange):
    four = input.swapaxes(0,1)
    #pspec = np.zeros((nrange,nc))
    new_pspec = np.zeros((nrange,nc))
    #for k in range(nsets):
    #    pspec += four[:,k*nc:(k+1)*nc]

    for l in range(nrange):
        if l == 0:
            new_pspec[0] = four[0]
        elif(l<nrange-1):
            new_pspec[l]=(four[l-1]+four[l]+four[l+1])/3
        else:
            new_pspec[l]=four[l]

    prom_pspec= np.mean(new_pspec,1)
    noise = np.median(prom_pspec[0:200])

    return noise

def Noise_RGB(data_spc,threshv=0.167):
    s        = data_spc.transpose()
    L        = s.shape[0] # Number of profiles
    N        = s.shape[1] # Number of heights

    data_RGB =np.zeros([3,N])
    im=int(math.floor(L/2)) #50
    i0l=im - int(math.floor(L*threshv)) #10
    i0h=im + int(math.floor(L*threshv)) #90
    ncount=0.0

    for ri in np.arange(N):
        data_RGB[0,ri] = np.sum(s[0:i0l,ri])
        data_RGB[1,ri] = np.sum(s[i0l:i0h,ri])
        data_RGB[2,ri] = np.sum(s[i0h:L,ri])

    image    = data_RGB.transpose()
    noise    = list(range(3))
    r2       = min(1000,image.shape[0]-1)
    r1       = int(r2*0.9)
    npy      = image.shape[1]
    
    noise[0]=noise[1]=noise[2]=0.0

    for i in range(r1,r2):
        ncount += 1
        for j in range(npy):
            noise[j]+=image[i,j]
    
    for j in range(npy):
        noise[j]/=ncount

    return noise

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

def freq_to_spec(freq):

    pass
def spec_to_freq(spec):
    #detectar si es convexo o concavo
    power = np.fft.ifftshift(spec.T,axes=0)
    return power

def light_fil(data):
    alturas= data.shape[0]
    freqs= data.shape[1]
    new_pspec = np.zeros((alturas,freqs))
    #print("SIZE!",size)
    for l in range(alturas):
        if l == 0:
            new_pspec[0] = data[0]
        elif (l< alturas-1):
            new_pspec[l]= (data[l-1]+data[l]+data[l+1])/3
        else:
            new_pspec[l]=data[l]
    del alturas
    del freqs
    return new_pspec


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

def Guardado_reducted(path,pw_ch0,pw_ch1,noise_a,noise_b,min_a,min_b):
    print("PATH GUARDADO: ",path)
    folder = path.split("/")[-2]
    code = folder[2]
    name = path.split("/")[-1][:-5]
    path_o = path[:-30]
    #day = 

    f = h5py.File(path,'r')
    #cspec01 = np.array(f['cspec01_C%s'%(code)])
    #dc0 = np.array(f['dc0_C%s'%(code)])
    #dc1 = np.array(f['dc1_C%s'%(code)])
    #image0 = np.array(f['image0_C%s'%(code)])
    #image1 = np.array(f['image1_C%s'%(code)])
    tiempo = np.array(f['t'])
    f.close()

    with h5py.File('%s.hdf5'%(path_o+folder),'a') as f:
    #with h5py.File('d%s.hdf5'%(day),'a') as f:

        
        g = f.create_group('%s'%(name))
        g.create_dataset('pw0_C%s'%(code),data=pw_ch0)
        g.create_dataset('noise0_C%s'%(code),data=noise_a)
        g.create_dataset('min0_C%s'%(code),data=min_a)

        g.create_dataset('pw1_C%s'%(code),data=pw_ch1)
        g.create_dataset('noise1_C%s'%(code),data=noise_b)
        g.create_dataset('min1_C%s'%(code),data=min_b)
        g.create_dataset('t',data=tiempo)

        #g.create_dataset('cspec01_C%s'%(code),data= cspec01)
        #g.create_dataset('dc0_C%s'%(code),data= dc0)
        #g.create_dataset('dc1_C%s'%(code),data= dc1)
        #g.create_dataset('image0_C%s'%(code),data= image0)
        #g.create_dataset('image1_C%s'%(code),data= image1)
    
    print("Guardado hdf5: ",'%s.hdf5 - %s'%(folder,name))
    print(" ")

def Guardado_same(path, path_o, pw_ch0,pw_ch1,Noise_a,Noise_b,Noise_RGB_0,Noise_RGB_1):
    
    folder = path.split("/")[-2]
    code = folder[2]
    name = path.split("/")[-1][:-5]

    #path_o = path[:-30]

    g = h5py.File(path,'r')
    tiempo = np.array(g['t'])
    g.close()

    if path[:-22] == path_o:
        with h5py.File('%s'%(path),'r+') as f:
        
            try:
            
                del f['pw0_C%s'%(code)]
                del f['pw1_C%s'%(code)]
                del f['noise0_C%s'%(code)]
                del f['noise1_C%s'%(code)]

            except:
                f.create_dataset('pw0_C%s'%(code),data = pw_ch0)
                f.create_dataset('pw1_C%s'%(code),data = pw_ch1)
                f.create_dataset('noise0_C%s'%(code),data =Noise_a)
                f.create_dataset('noise1_C%s'%(code),data =Noise_b)
                #f.create_dataset('noiseRGB0_C%s'%(code),data = Noise_RGB_0)
                #f.create_dataset('noiseRGB1_C%s'%(code),data = Noise_RGB_1)

    else:
        print(path_o)
        try:
            os.makedirs(path_o)
        except FileExistsError:
            print("Archivo ya existente",'%s.hdf5'%(path_o+name))

        with h5py.File('%s.hdf5'%(path_o+name),'w') as f:

            f.create_dataset('pw0_C%s'%(code),data = pw_ch0)
            f.create_dataset('pw1_C%s'%(code),data = pw_ch1)
            f.create_dataset('noise0_C%s'%(code),data =Noise_a)
            f.create_dataset('noise1_C%s'%(code),data =Noise_b)
            f.create_dataset('noiseRGB0_C%s'%(code),data = Noise_RGB_0)
            f.create_dataset('noiseRGB1_C%s'%(code),data = Noise_RGB_1)
            f.create_dataset('t',data=tiempo)

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

    pw_ch0 = spec_to_freq(FS_filtrado_A)
    #pw_ch0 = np.fft.ifftshift(FS_filtrado_A.T,axes=0)

    pw_ch1 = spec_to_freq(FS_filtrado_B)
    #pw_ch1 = np.fft.ifftshift(FS_filtrado_B.T,axes=0)
    

    plt.figure(figsize=(20,4))
    plt.subplot(1,4,1)
    plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_a), cmap='jet')
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.title("%s - %s"%(tiempo,"CH0-ori"))
      
    plt.subplot(1,4,2)
    plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch0,axes=0).T), cmap='jet')
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
    plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch1,axes=0).T), cmap='jet')
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


def lectura_matrix_sparse(file):
    
    
    pass

print ("*** Inicinado PROC - FILTRADO ***")
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
########################## REDUCTED DATA ### 600 o 100 perfiles ###############################################################################
parser.add_argument('-R',action='store',dest='reducted',type=int,help='Reduccion de almacenamiento. Data sparse. Por defecto, se esta ingresando 0',default=0)

parser.add_argument('-clean',action='store',dest='clean',type=int,help='Almacenar matriz filtrada sin reducir. Por defecto, se esta ingresando 0',default=0)
########################## PLOT DATA ### 600 o 100 perfiles ###############################################################################
parser.add_argument('-P',action='store',dest='plot',type=int,help='Almacenamiento de data ploteada. Potencia original vs Potencia filtrada. Por defecto, se esta ingresando 0',default=0)

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

parser.add_argument('-path_o',action='store',dest='path_out',help='Directorio de Datos \
					.Por defecto, se esta ingresando entre comillas /media/soporte/PROCDATA/',default='/media/soporte/PROCDATA/')
#Parsing the options of the script
results	   = parser.parse_args()
path	   = str(results.path_lectura) #/media/igp-114/PROCDATA/
path_o     = str(results.path_out)
freqs	   = results.f_freq            # 2.72216796875
campaign   = results.c_campaign        # Mode 0(normal) y 1(Campaign)
clean      = results.clean
plot       = results.plot
code	   = int(results.code_seleccionado)
Days	   = results.date_seleccionado
lo		   = results.lo_seleccionado
reducted   = int(results.reducted)
graphics_folder = results.graphics_folder

if campaign == 1:
    path = path + "CAMPAIGN/"
    #path= "/home/soporte/Spectros/CAMPAIGN/"
    path_o = path_o + "CAMPAIGN/"
    nc = 600
else:
    path = path
    path_o = path_o
    nc = 100

if freqs <3:
    ngraph = 0
else:
    ngraph = 1

from datetime import datetime
days = datetime.strptime(Days, "%Y/%m/%d")
global day
day = days.strftime("%Y%j")
#print(day)

folder = "sp%s1_f%s"%(code,ngraph)
path     = path+"d"+day+'/'+folder
graphics_folder = graphics_folder+"d"+day+'/'+folder+"/"
path_o   = path_o+"d"+day+'/'+folder+'/'

print ("Path:",path)
#------------------------------------------------------------------------------------------------------
nrange=1000
code = code
freq = float("%se6"%(freqs))

files = glob.glob(path+"/spec-*.hdf5")
files.sort()
print(folder)
#print(files)

for CurrentSpec in files:
    print("Archivo",CurrentSpec)
    f = h5py.File(CurrentSpec,'r')
    name = CurrentSpec.split("/")[-1][5:-5]
    tiempo = str(datetime.fromtimestamp(int(name)))
    print("NAME:",name," -- ",tiempo)
    
    ch0 = np.array(f['pw0_C%s'%(code)])
    ch1 = np.array(f['pw1_C%s'%(code)])
    print("FORMA:::",ch0.shape)
    ## Ligero Filtrado 
    #ch0 = light_fil(ch0.T).T
    #ch1 = light_fil(ch1.T).T
    ##
    Noise_a = Ruido(ch0,nc,nrange)
    Noise_b = Ruido(ch1,nc,nrange)
    
    NoiseRGB_a = Noise_RGB(ch0,0.25)
    NoiseRGB_b = Noise_RGB(ch1,0.25)
    #print("")
    #print("NoiseRGB_a",NoiseRGB_a)
    #print("NoiseRGB_b",NoiseRGB_b)

    FullSpectra_a = np.fft.fftshift(ch0,axes=0).T
    FullSpectra_b = np.fft.fftshift(ch1,axes=0).T

    f.close()

    Freqs = np.linspace(-5,5,nc)
    Ranges = np.linspace(0,1500,nrange)

    #min_a = np.min(FullSpectra_a)
    #min_b = np.min(FullSpectra_b)
    
    NoiseFloor_a = np.median( FullSpectra_a, axis=0)
    NoiseFloor_b = np.median( FullSpectra_b, axis=0)
    #nc = 1000 alturas

    aux_a =  np.reshape(NoiseFloor_a, (1,nc))
    aux_b =  np.reshape(NoiseFloor_b, (1,nc))

    FullSpec_clean_a = np.log10(FullSpectra_a) - np.log10(aux_a)
    FullSpec_clean_a = np.where(FullSpec_clean_a<=0.0, 0, FullSpec_clean_a)

    FullSpec_clean_b = np.log10(FullSpectra_b) - np.log10(aux_b)
    FullSpec_clean_b = np.where(FullSpec_clean_b<=0.0, 0, FullSpec_clean_b)

    # Limpieza de bandas
    FullSpec_clean_a = delete_bands(5,FullSpec_clean_a)
    FullSpec_clean_b = delete_bands(5,FullSpec_clean_b)

    #### Uso del metodo estadìstico percentil 99 o 98
    auxperc_a = np.percentile(FullSpec_clean_a, q=98, axis=(0,1))
    auxperc_b = np.percentile(FullSpec_clean_b, q=98, axis=(0,1))
    #print("AUX_PERC: ", auxperc_a)
    #print("AUX_PERC: ", auxperc_b)

    Pot_a = (auxperc_a<=FullSpec_clean_a)*1
    Pot_b = (auxperc_b<=FullSpec_clean_b)*1

    for i in range(nrange):
        for j in range(nc):
            if Pot_a[i][j] == 0:
                #FullSpectra_a[i][j] = min_a
                FullSpec_clean_a[i][j] = 0

            if Pot_b[i][j] == 0:
                #FullSpectra_b[i][j] = min_b
                FullSpec_clean_b[i][j] = 0

    #Creacion de la data de solo componentes de potencia canal 0 sin valores iguales a cero
    sparse_matrix_a = sparse.csr_matrix(FullSpec_clean_a).tocoo()
    #Creacion de la data de solo componentes de potencia canal 1 sin valores iguales a cero
    sparse_matrix_b = sparse.csr_matrix(FullSpec_clean_b).tocoo()
    
    ###Almacenando en un DataFrame para un facil manejo al eliminar etiquetas con ruido
    #Creacion de la data de solo componentes de canal 0
    Frecuencia_a = sparse_matrix_a.col
    Rango_a = sparse_matrix_a.row
    data_a = pd.DataFrame()
    data_a['Frecuencia']=Frecuencia_a
    data_a['Rango']=Rango_a
    X_a = data_a.values

    #Creacion de la data de solo componentes de canal 1
    Frecuencia_b = sparse_matrix_b.col
    Rango_b = sparse_matrix_b.row
    data_b = pd.DataFrame()
    data_b['Frecuencia']=Frecuencia_b
    data_b['Rango']=Rango_b
    X_b = data_b.values

    eps_a = epsilon(X_a, vecinos = 10)
    eps_b = epsilon(X_b, vecinos = 10)

    db_a = DBSCAN(eps=eps_a, min_samples = 10, metric= "euclidean").fit(X_a)
    cluster_a = db_a.labels_
    data_a['label'] = cluster_a
    
    db_b = DBSCAN(eps=eps_b, min_samples = 10, metric= "euclidean").fit(X_b)
    cluster_b = db_b.labels_
    data_b['label'] = cluster_b
    
    ### Eliminacion de ruido
    data_a = data_a.drop(data_a[data_a['label'] == -1].index)
    data_b = data_b.drop(data_b[data_b['label'] == -1].index)

    # Dataframe "limpio"
    f1_a = data_a['Frecuencia'].values
    f2_a = data_a['Rango'].values

    # Dataframe "limpio"
    f1_b = data_b['Frecuencia'].values
    f2_b = data_b['Rango'].values
    #print("F1_b",type(f1_b),f2_b.shape)

    New_Full_Spectra_a = []
    New_Full_Spectra_b = []

    ch0=light_fil(spec_to_freq(FullSpectra_a).T).T
    ch1=light_fil(spec_to_freq(FullSpectra_b).T).T

    FullSpectra_a = np.fft.fftshift(ch0,axes=0).T
    FullSpectra_b = np.fft.fftshift(ch1,axes=0).T
    #print("LAST FORM",FullSpectra_a.shape)

    min_a = np.min(FullSpectra_a)
    min_b = np.min(FullSpectra_b)

    for k in range(len(f1_a)):
    
        New_Full_Spectra_a.append(FullSpectra_a[f2_a[k]][f1_a[k]])

    New_Full_Spectra_a = np.array(New_Full_Spectra_a) 
    #print("New_Full_Spectra_a",New_Full_Spectra_a)
    
    for k in range(len(f1_b)):

        New_Full_Spectra_b.append(FullSpectra_b[f2_b[k]][f1_b[k]])

    New_Full_Spectra_b = np.array(New_Full_Spectra_b)

    ### Resultado de la matrix B sin ruido
    New_Full_Spectra_a = csr_matrix((New_Full_Spectra_a,(f2_a,f1_a)),shape=(1000,nc))
    #ncomp_a = New_Full_Spectra_a.tocoo()
    #Pow_reduc_a = np.zeros((len(ncomp_a.col),3))
    #Pow_reduc_a[:,0],Pow_reduc_a[:,1],Pow_reduc_a[:,2] = ncomp_a.col, ncomp_a.row, ncomp_a.data
    #print("   Guardando ch0: ")

    ### Resultado de la matrix A sin ruido
    New_Full_Spectra_b = csr_matrix((New_Full_Spectra_b,(f2_b,f1_b)),shape=(1000,nc))
    #ncomp_b = New_Full_Spectra_b.tocoo()
    #Pow_reduc_b = np.zeros((len(ncomp_b.col),3))
    #Pow_reduc_b[:,0],Pow_reduc_b[:,1],Pow_reduc_b[:,2] = ncomp_b.col, ncomp_b.row, ncomp_b.data
    #print("   Guardando ch1: ")

    if reducted == 1:
        
        Spectra_a =New_Full_Spectra_a.toarray()
        #Spectra_a = np.fft.ifftshift(Spectra_a.T,axes=0)
        Spectra_a = spec_to_freq(Spectra_a)
        ncomp_a = csr_matrix(Spectra_a).tocoo()
        #ncomp_a = Spectra_a.tocoo()
        Pow_reduc_a = np.zeros((len(ncomp_a.col),3))
        Pow_reduc_a[:,0],Pow_reduc_a[:,1],Pow_reduc_a[:,2] = ncomp_a.col, ncomp_a.row, ncomp_a.data


        Spectra_b = New_Full_Spectra_b.toarray()
        #Spectra_b = np.fft.ifftshift(Spectra_b.T,axes=0)
        Spectra_b = spec_to_freq(Spectra_b)
        ncomp_b   = csr_matrix(Spectra_b).tocoo()
        Pow_reduc_b = np.zeros((len(ncomp_b.col),3))
        Pow_reduc_b[:,0],Pow_reduc_b[:,1],Pow_reduc_b[:,2] = ncomp_b.col, ncomp_b.row, ncomp_b.data

        #Guardado_reducted(CurrentSpec,Pow_reduc_a,Pow_reduc_b,Noise_a,Noise_b, min_a, min_b)
        Guardado_reducted(CurrentSpec,Pow_reduc_a,Pow_reduc_b,Noise_a,Noise_b, min_a, min_b)

        Noise_a
        Noise_b
        del Spectra_a
        del Spectra_b

    if clean == 1:
        #New_Full_Spectra_a = csr_matrix((New_Full_Spectra_a,(f2_a,f1_a)),shape=(1000,nc))


        FS_filtrado_A = New_Full_Spectra_a.toarray()
        #FS_filtrado_A[FS_filtrado_A == 0] = min_a
        FS_filtrado_A[FS_filtrado_A == 0] = Noise_a   ### Agregado el reemplazo del min_a por  Noise_a
        #pw_ch0 = np.fft.ifftshift(FS_filtrado_A.T,axes=0)
        pw_ch0 = spec_to_freq(FS_filtrado_A)
        ##

        #New_Full_Spectra_b = csr_matrix((New_Full_Spectra_b,(f2_b,f1_b)),shape=(1000,nc))
    
        FS_filtrado_B = New_Full_Spectra_b.toarray()
        #FS_filtrado_B[FS_filtrado_B == 0] = min_b
        FS_filtrado_B[FS_filtrado_B == 0] = Noise_b   ### Agregado el reemplazo del min_b por  Noise_b
        #pw_ch1 = np.fft.ifftshift(FS_filtrado_B.T,axes=0)
        pw_ch1 = spec_to_freq(FS_filtrado_B)

        print("CurrentSpec!",CurrentSpec)
        print("PATH_O:",path_o)
        Guardado_same(CurrentSpec, path_o, pw_ch0,pw_ch1,Noise_a,Noise_b,NoiseRGB_a,NoiseRGB_b)


        #Guardado_same(path, pw_ch0,pw_ch1,Noise_a,Noise_b)
        del FS_filtrado_A
        del FS_filtrado_B

    if plot == 1:
        print("** PLOTEO **")
        #New_Full_Spectra_a = csr_matrix((New_Full_Spectra_a,(f2_a,f1_a)),shape=(1000,nc))
        #New_Full_Spectra_b = csr_matrix((New_Full_Spectra_b,(f2_b,f1_b)),shape=(1000,nc))

        #ploteado(New_Full_Spectra_a,min_a,New_Full_Spectra_b,min_b,FullSpectra_a,FullSpectra_b,name,graphics_folder,eps_a,eps_b)
        #ploteado(New_Full_Spectra_a,Noise_a,New_Full_Spectra_b,Noise_b,FullSpectra_a,FullSpectra_b,name,graphics_folder,eps_a,eps_b)
        
    #else:
    #    continue

##### *** PARA PLOTEAR  *** #####
'''   
    FS_filtrado_A = New_Full_Spectra_a.toarray()
    FS_filtrado_A[FS_filtrado_A == 0] = min_a


    FS_filtrado_B = New_Full_Spectra_b.toarray()
    FS_filtrado_B[FS_filtrado_B == 0] = min_b

    pw_ch0 = np.fft.ifftshift(FS_filtrado_A.T)
    pw_ch1 = np.fft.ifftshift(FS_filtrado_B.T)

    plt.figure(figsize=(20,4))
    plt.subplot(1,4,1)
    print("BEFORE PLOTEO,",FullSpectra_a.shape)
    plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_a), cmap='jet')
    plt.colorbar()
    plt.subplot(1,4,2)
    plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch0).T), cmap='jet')
    plt.colorbar()
    plt.subplot(1,4,3)
    plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_b), cmap='jet')
    plt.colorbar()
    plt.subplot(1,4,4)
    plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch1).T), cmap='jet')
    plt.colorbar()

    #plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch1).T), cmap='jet')
    #plt.colorbar()

    plt.show()
    #plt.pause(1)
'''