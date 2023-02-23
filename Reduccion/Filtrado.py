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
import warnings
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
import math,time,os,sys,shutil
import _noise

#Almacenar Ploteo de datos filtrados
#./Filtrado.py -f 2.72 -code 0 -C 1 -date "2022/08/29" -P 1 
#./Filtrado.py -f 2.72 -code 0 -C 1 -date "2022/08/29" -P 1 -graphics_folder "/home/soporte/Pictures/"

#Almacenar matriz de potencias reducidas (data sparse)
#./Filtrado.py -f 2.72 -code 2 -C 1 -date "2022/08/29" -R 1
#./Filtrado.py -f 2.72 -code 2 -C 1 -date "2022/08/29" -h 0 -R 1 -path '/media/soporte/RAWDATA/HYOa/'

#Reducir y ver plots
#./Filtrado.py -f 2.72 -code 2 -C 1 -date "2022/08/29" -R 1 -P 1

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

def Ruido_v2(matrix_ch,nc,nrange):
    
    Freqs        = np.linspace(-5,5,nc)
    Ranges       = np.linspace(0,1500,nrange)
    indice_menor = int(nc/2)-int(math.floor(nc*0.1))
    indice_mayor = int(nc/2)+int(math.floor(nc*0.1))
    #print("indice_menor",indice_menor)
    #print("indice_mayor",indice_mayor)
    #Seccion de Matriz adecuada para la deteccion de Ruido
    ruido_new    = (matrix_ch)[:201,indice_menor:indice_mayor]  
    prom_pspec   = np.median(ruido_new)
    return prom_pspec


def hildebrand_sekhon(input,int_incoh):
    noise = _noise.hildebrand_sekhon(input,int_incoh)
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
        eps = 6
        print("EPS EDITADO CON KNEED:",eps,end=" - ")
    elif eps<4:
        eps= 4.3
        print("EPS EDITADO CON KNEED:",eps,end=" - ")
    else:
        print("EPS CALCULADO CON KNEED:",eps,end=" - ")
    return eps

def DBSCAN_algorythm(matrix,FullSpectra,vecinos):
    #Creacion de data sparse (Solo componentes != 0).
    sparse_matrix      = sparse.csr_matrix(matrix).tocoo()
    Frecuencia         = sparse_matrix.col
    Rango              = sparse_matrix.row
    #Colocacion de datos en DataFrame.
    data               = pd.DataFrame()
    data['Frecuencia'] = Frecuencia
    data['Rango']      = Rango
    X_pow              = data.values
    #Hallar el epsilon de agrupaciones.
    eps           = epsilon(X_pow, vecinos = vecinos)
    #Ejecucion de DBSCAN.
    db            = DBSCAN(eps=eps, min_samples = vecinos, metric= "euclidean").fit(X_pow)
    clusters      = db.labels_
    data['label'] = clusters
    #Eliminación de etiquetas de ruido.
    data          = data.drop(data[data['label'] == -1].index)
    #Dataframe limpio
    f1_a          = data['Frecuencia'].values
    f2_a          = data['Rango'].values

    New_Full_Spectra = []

    for k in range(len(f1_a)):
    
        New_Full_Spectra.append(FullSpectra[f2_a[k]][f1_a[k]])
    
    New_Full_Spectra = np.array(New_Full_Spectra)
    New_Full_Spectra = csr_matrix((New_Full_Spectra,(f2_a,f1_a)),shape=(1000,nc))
    
    return New_Full_Spectra

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

def removeDC(FullSpectra,nc,nrange, mode=2):

    jspectra  = FullSpectra.reshape(1,nc,nrange)
    num_chan  = jspectra.shape[0]
    num_hei   = jspectra.shape[2]
    freq_dc   = int(jspectra.shape[1] / 2)
    ind_vel   = np.array([-2, -1, 1, 2]) + freq_dc
    ind_vel   = ind_vel.astype(int)

    if ind_vel[0] < 0:
        ind_vel[list(range(0, 1))] = ind_vel[list(range(0, 1))] + self.num_prof

    if mode == 1:
        jspectra[:, freq_dc, :] = (jspectra[:, ind_vel[1], :] + jspectra[:, ind_vel[2], :]) / 2  # CORRECCION

    if mode == 2:
        vel = np.array([-2, -1, 1, 2])
        xx = np.zeros([4, 4])

        for fil in range(4):
            xx[fil, :] = vel[fil]**np.asarray(list(range(4)))

        xx_inv = np.linalg.inv(xx)
        xx_aux = xx_inv[0, :]

        for ich in range(num_chan):
            yy = jspectra[ich, ind_vel, :]
            jspectra[ich, freq_dc, :] = np.dot(xx_aux, yy)
            junkid = jspectra[ich, freq_dc, :] <= 0
            cjunkid = sum(junkid)
            if cjunkid.any():
                jspectra[ich, freq_dc, junkid.nonzero()] = (jspectra[ich, ind_vel[1], junkid] + jspectra[ich, ind_vel[2], junkid]) / 2

    return jspectra[0]



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

def delete_folder(path,files):
    #print("Funcion ",path+'.hdf5')
    f   = h5py.File(path+'.hdf5','r')
    filelist = sorted(list(f.keys()))
    f.close()
    
    if filelist[-1] == files[-1].split("/")[-1][:-5]:
        print("....Borrando")
        print("")
        print("    FOLDER ELIMINADO:",path)
        shutil.rmtree("%s"%(path))

def Guardado_reducted(path,path_o,pw_ch0,pw_ch1,noise_a,noise_b):
    folder = path.split("/")[-2]
    print("\n PATH GUARDADO: ",path)
    code = folder[2]
    name = path.split("/")[-1][:-5]


    f = h5py.File(path,'r')
    #cspec01 = np.array(f['cspec01_C%s'%(code)])
    #dc0 = np.array(f['dc0_C%s'%(code)])
    #dc1 = np.array(f['dc1_C%s'%(code)])
    #image0 = np.array(f['image0_C%s'%(code)])
    #image1 = np.array(f['image1_C%s'%(code)])
    tiempo = np.array(f['t'])
    f.close()
    #print("HOLI:",'%s.hdf5'%(path_o+folder))
    try:
        os.makedirs(path_o)
    except FileExistsError:
            print("Archivo ya existente",path_o)

    with h5py.File('%s.hdf5'%(path_o+folder),'a') as f:
    #with h5py.File('d%s.hdf5'%(day),'a') as f:

        g = f.create_group('%s'%(name))
        g.create_dataset('pw0_C%s'%(code),data=pw_ch0)
        g.create_dataset('noise0_C%s'%(code),data=noise_a)
        #g.create_dataset('min0_C%s'%(code),data=min_a)

        g.create_dataset('pw1_C%s'%(code),data=pw_ch1)
        g.create_dataset('noise1_C%s'%(code),data=noise_b)
        #g.create_dataset('min1_C%s'%(code),data=min_b)
        g.create_dataset('t',data=tiempo)

        #g.create_dataset('cspec01_C%s'%(code),data= cspec01)
        #g.create_dataset('dc0_C%s'%(code),data= dc0)
        #g.create_dataset('dc1_C%s'%(code),data= dc1)
        #g.create_dataset('image0_C%s'%(code),data= image0)
        #g.create_dataset('image1_C%s'%(code),data= image1)
    
    print(" Guardado hdf5: ",'%s.hdf5 - %s'%(folder,name))
    #print(" ")

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

def ploteado(New_Full_Spectra_a,noise_a,New_Full_Spectra_b,noise_b,FullSpectra_a,FullSpectra_b,name,graphics_folder):
    import datetime
    tiempo = str(datetime.datetime.fromtimestamp(int(name)))
    print("Graphics_folder",graphics_folder)
    ##
    min_value_a =np.min(FullSpectra_a)
    max_value_a = np.max(FullSpectra_a)
    
    min_value_b =np.min(FullSpectra_b)
    max_value_b = np.max(FullSpectra_b)
    ####
    FS_filtrado_A = New_Full_Spectra_a.toarray()
    FS_filtrado_A[FS_filtrado_A == 0] = noise_a

    FS_filtrado_B = New_Full_Spectra_b.toarray()
    FS_filtrado_B[FS_filtrado_B == 0] = noise_b

    #pw_ch0 = np.fft.ifftshift(FS_filtrado_A)
    #pw_ch1 = np.fft.ifftshift(FS_filtrado_B)

    pw_ch0 = spec_to_freq(FS_filtrado_A)
    #pw_ch0 = np.fft.ifftshift(FS_filtrado_A.T,axes=0)

    pw_ch1 = spec_to_freq(FS_filtrado_B)
    #pw_ch1 = np.fft.ifftshift(FS_filtrado_B.T,axes=0)   

    plt.figure(figsize=(20,4))
    plt.subplot(1,4,1)
    #plt.pcolor(Freqs, Ranges, np.log10(FullSpectra_a), cmap='jet',shading='auto')
    plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_a), cmap='jet', shading='auto',vmin=np.log10(min_value_a),vmax=np.log10(max_value_a))
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.ylim((0,1500))
    plt.title("%s - %s"%(tiempo,"CH0-ori"))
      
    plt.subplot(1,4,2)
    #plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch0,axes=0).T), cmap='jet')
    plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch0,axes=0).T), cmap='jet',vmin=np.log10(min_value_a),vmax=np.log10(max_value_a))
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.ylim((0,1500))
    plt.title("%s - %s. "%(tiempo,"CH0-Filt"))
    
    plt.subplot(1,4,3)
    #plt.pcolor(Freqs, Ranges, np.log10(FullSpectra_b), cmap='jet',shading='auto')
    plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_b), cmap='jet',vmin=np.log10(min_value_b),vmax=np.log10(max_value_b))
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.ylim((0,1500))
    plt.title("%s - %s"%(tiempo,"CH1-ori"))
    
    plt.subplot(1,4,4)
    #plt.pcolor(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch1,axes=0).T), cmap='jet',shading='auto')
    plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch1,axes=0).T), cmap='jet',vmin=np.log10(min_value_b),vmax=np.log10(max_value_b))
    plt.colorbar()
    plt.xlabel("Frecuencia")
    plt.ylabel("Alturas")
    plt.ylim((0,1500))
    plt.title("%s - %s. "%(tiempo,"CH1-Filt"))
    try:
        plt.savefig(fname=graphics_folder+name+".png",dpi=200)
    except:
        os.makedirs(graphics_folder)
        print(" Directed %s created"%(graphics_folder))
        plt.savefig(fname=graphics_folder+name+".png",dpi=200)
    
    #plt.savefig('/home/soporte/Pictures/d2022241/sp01_f0/hola.png')
    plt.close()


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
        array[array == 0] = read_newdata(path,filename,nc,'noise'+str(value[2])+'_C'+str(code))
    else:
        array = fp.get(dir_file)[()]
    fp.close()
    return array
    
def plot_filtdata(path_o,folder,nc,code,graphics_folder):
    import datetime
    try:
        f        = h5py.File(path_o+folder+'.hdf5','r')
        filelist = sorted(list(f.keys()))
        f.close()
    except OSError:
        print("    NO EXISTE EL ARCHIVO DE LECTURA:",path+'.hdf5')
        exit()
    nrange= 1000

    for filename in filelist:
        #print("FILENAME",filename)
        tiempo = str(datetime.datetime.fromtimestamp(int(filename[5:])))
        #print("FILENAME",time)
        pwch0 = read_newdata(path_o+folder,filename,nc,value='pw'+str(0)+'_C'+str(code))
        pwch1 = read_newdata(path_o+folder,filename,nc,value='pw'+str(1)+'_C'+str(code))

        Freqs = np.linspace(-5,5,nc)
        Ranges = np.linspace(0,1500,nrange)

        FullSpectra_a = np.fft.fftshift(pwch0,axes=0).T
        FullSpectra_b = np.fft.fftshift(pwch1,axes=0).T

        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_a), cmap='jet')
        plt.colorbar()
        plt.xlabel("Frecuencia")
        plt.ylabel("Alturas")
        plt.title("%s - %s"%(tiempo,"CH0-filt"))

        plt.subplot(1,2,2)
        plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_a), cmap='jet')
        plt.colorbar()
        plt.xlabel("Frecuencia")
        plt.ylabel("Alturas")
        plt.title("%s - %s"%(tiempo,"CH1-filt"))
        #plt.show()
        #try:
        plt.savefig(fname=graphics_folder+filename+".png",dpi=200)
        #except:
        #    os.makedirs(graphics_folder)
        #    print(" Directed %s created"%(graphics_folder))
        #    plt.savefig(fname=graphics_folder+filename+".png",dpi=200)
        
        


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
########################## Ruido Hildebrand-seckon ####################################################################################################
parser.add_argument('-h_s',action='store',dest='hild_sk',type=int,help='Obtener Ruido con Hildebrand_sekhon ',default=0)
########################## CAMPAIGN ### 600 o 100 perfiles ###############################################################################
parser.add_argument('-C',action='store',dest='c_campaign',type=int,help='Campaign 1 (600 perfiles) y 0(100 perfiles). Por defecto, se esta ingresando 1',default=1)
########################## REDUCTED DATA ### 600 o 100 perfiles ###############################################################################
parser.add_argument('-R',action='store',dest='reducted',type=int,help='Reduccion de almacenamiento. Data sparse. Por defecto, se esta ingresando 0',default=0)

parser.add_argument('-clean',action='store',dest='clean',type=int,help='Almacenar matriz filtrada sin reducir. Por defecto, se esta ingresando 0',default=0)
########################## PLOT DATA COMPARED### 600 o 100 perfiles ###############################################################################
#parser.add_argument('-P_c',action='store',dest='plot_c',type=int,help='Almacenamiento de comparacion de data ploteada. Potencia original vs Potencia filtrada.\
#                    Por defecto, se esta ingresando 0',default=0)

########################## PLOT DATA FILTRADA### 600 o 100 perfiles ###############################################################################
parser.add_argument('-P',action='store',dest='plot',type=int,help='Almacenamiento de comparacion de data ploteada. Potencia original vs Potencia filtrada.\
                    Por defecto, se esta ingresando 0',default=0)

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
########################## GRAPHICS - RESULTS ###################################################################################################
parser.add_argument('-graphics_folder',action='store',dest='graphics_folder',help='Directorio de Resultados \
					.Por defecto, se esta ingresando entre comillas /home/soporte/Pictures/', default='/home/soporte/Pictures/')

parser.add_argument('-path_o',action='store',dest='path_out',help='Directorio de Datos \
					.Por defecto, se esta ingresando entre comillas /media/soporte/PROCDATA/',default='/media/soporte/PROCDATA/')

##########################  REMOVE DC ############################################################################################################
parser.add_argument('-dc',action='store',dest='remove_dc',help='Argumento para eliminar la señal DC \
		             de un espectro',default=0)
########################## DELETE SPECTRA FILES ###################################################################################################
parser.add_argument('-del',action='store',dest='delete',type=int,help='Borrado de data espectral. Data sparse. Por defecto, se esta ingresando 0',default=0)
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
lo		   = str(results.lo_seleccionado)
reducted   = int(results.reducted)
graphics_folder = results.graphics_folder
deleted    = int(results.delete)
hild_sk    = results.hild_sk
remove_dc  = int(results.remove_dc)

if campaign == 1:
    path      = path + "CAMPAIGN/"
    #path     = "/home/soporte/Spectros/CAMPAIGN/"
    path_o    = path_o + "CAMPAIGN/"
    nc        = 600
    int_incoh = 6
else:
    path      = path
    path_o    = path_o
    nc        = 100
    int_incoh = 1

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
path_o   = path_o+"d"+day+'/'

print("Path:",path)
#print("Path:",'%s.hdf5'%(path+folder))
#print("Existe:",os.path.exists('%s.hdf5'%(path)))

files = glob.glob(path+"/spec-*.hdf5")
files.sort()
print(folder)

#if plot == 1:
#    print("Plot - lectura de datos filtrados")
#    #plot_filtdata(path_o,folder,nc,code,graphics_folder)
#    print("Espectros filtrados en:",graphics_folder)
    
if os.path.exists('%s.hdf5'%(path)):
    os.system('rm -r %s.hdf5'%(path))
    
#------------------------------------------------------------------------------------------------------
nrange=1000
code = code
freq = float("%se6"%(freqs))

for CurrentSpec in files:
    print("*Archivo",CurrentSpec)
    try:
        f = h5py.File(CurrentSpec,'r')
    except OSError:
        print("\nError en el espectro *.hdf5\n")
        continue

    name = CurrentSpec.split("/")[-1][5:-5]
    tiempo = str(datetime.fromtimestamp(int(name)))
    print(" TIME:",tiempo,end=" ")
    
    try:
        ch0 = np.array(f['pw0_C%s'%(code)])
    except KeyError:
        print("Espectro vacio \n")
        continue

    try:
        ch1 = np.array(f['pw1_C%s'%(code)])
    except KeyError:
        print("Espectro vacio \n")
        continue

    print("FORMA ch0:::",ch0.shape)
   
    NoiseRGB_a = Noise_RGB(ch0,0.25)
    NoiseRGB_b = Noise_RGB(ch1,0.25)
    #print("")
    #print("NoiseRGB_a",NoiseRGB_a)
    #print("NoiseRGB_b",NoiseRGB_b)

    FullSpectra_a = np.fft.fftshift(ch0,axes=0).T
    FullSpectra_b = np.fft.fftshift(ch1,axes=0).T
    #print("FullSpectra",FullSpectra_a.shape)
    Freqs = np.linspace(-5,5,nc)
    Ranges = np.linspace(0,1500,nrange)
    
    #if lo == '21' or lo == '22':
    if remove_dc == 1:
        FullSpectra_a = removeDC(FullSpectra_a.T,nc,nrange, mode=2).T
        FullSpectra_b = removeDC(FullSpectra_b.T,nc,nrange, mode=2).T

    ch0=light_fil(spec_to_freq(FullSpectra_a).T).T
    ch1=light_fil(spec_to_freq(FullSpectra_b).T).T

    FullSpectra_a = np.fft.fftshift(ch0,axes=0).T
    FullSpectra_b = np.fft.fftshift(ch1,axes=0).T    

    ruido_a    = np.delete(FullSpectra_a, range(100,850), axis=0)
    ruido_b    = np.delete(FullSpectra_b, range(100,850), axis=0)
    #print("Forma de ruido:",ruido_a.shape)
    ruido_a_freq    = spec_to_freq(ruido_a)
    ruido_b_freq    = spec_to_freq(ruido_b)
    #print("Forma de ruido after:",ruido_a_freq.shape)

    if hild_sk == 1:
        print("\nRUIDO HILDEBRAND-SECKON\n")
        #Noise_a = hildebrand_sekhon(np.sort(ch0),int_incoh)
        Noise_a = hildebrand_sekhon(np.sort(ruido_a_freq),int_incoh)
        #Noise_b = hildebrand_sekhon(np.sort(ch1),int_incoh)
        Noise_b = hildebrand_sekhon(np.sort(ruido_b_freq),int_incoh)    
    else:
        print("RUIDO NORMAL")
        Noise_a = Ruido_v2(ch0.T,nc,nrange)
        Noise_b = Ruido_v2(ch1.T,nc,nrange)
        print("Noise_a:",Noise_a,"Noise_b:",Noise_b,"\n")   

    #NoiseFloor_a = np.median( FullSpectra_a, axis=0)
    NoiseFloor_a = np.median( ruido_a, axis=0)
    #print("NoiseFloor_a ",NoiseFloor_a.shape)
    #NoiseFloor_b = np.median( FullSpectra_b, axis=0)
    NoiseFloor_b = np.median( ruido_b, axis=0)

    #'Campaign':[percentil o porcentaje,vecinos] #porcentaje es IP a vecinos
    location = {'11':{'Campaign':{'ch0':[86,38],'ch1':[86,38]},'Normal':{'ch0':[92,15],'ch1':[93,16]}}, #JROA Good
                '12':{'Campaign':{'ch0':[86,38],'ch1':[86,38]},'Normal':{'ch0':[92,15],'ch1':[93,16]}}, #JROB Good
                '21':{'Campaign':{'ch0':[86,38],'ch1':[86,38]},'Normal':{'ch0':[91,30],'ch1':[95,18]}}, #HYOA Good
                '22':{'Campaign':{'ch0':[86,38],'ch1':[86,38]},'Normal':{'ch0':[92,28],'ch1':[95,18]}}, #HYOB
                '31':{'Campaign':[86,38],'Normal':[92,15]}, #Mala
                '41':{'Campaign':{'ch0':[88,38],'ch1':[86,38]},'Normal':{'ch0':[92,20],'ch1':[94,20]}}, #Merced
                '51':{'Campaign':{'ch0':[87,36],'ch1':[86,38]},'Normal':{'ch0':[93,19],'ch1':[95,18]}}, #Barranca Good
                '61':{'Campaign':[86,38],'Normal':[92,15]}, #Oroya                                       
                 }
    

    if nc == 100:
        NoiseFloor_a = np.median( FullSpectra_a, axis=0)
        NoiseFloor_b = np.median( FullSpectra_b, axis=0)
        #Porcentaje o percentil en Modo Normal
        percentil_a  = location[lo]['Normal']['ch0'][0]
        percentil_b  = location[lo]['Normal']['ch1'][0]
        #Vecinos del algorimo DBSCAN
        vecinos_a      = location[lo]['Normal']['ch0'][1]
        vecinos_b      = location[lo]['Normal']['ch1'][1]
    else:
        #Campaign Mode
        #NoiseFloor_a = np.median( FullSpectra_a, axis=0)
        #NoiseFloor_b = np.median( FullSpectra_b, axis=0)
        percentil_a  = location[lo]['Campaign']['ch0'][0]
        percentil_b  = location[lo]['Campaign']['ch1'][0]
        #Vecinos del algorimo DBSCAN
        vecinos_a      = location[lo]['Campaign']['ch0'][1]
        vecinos_b      = location[lo]['Campaign']['ch1'][1]

        
    f.close()
    aux_a =  np.reshape(NoiseFloor_a, (1,nc))
    
    aux_b =  np.reshape(NoiseFloor_b, (1,nc))

    FullSpec_clean_a = np.log10(FullSpectra_a) - np.log10(aux_a)
    FullSpec_clean_a = np.where(FullSpec_clean_a<=0.0, 0, FullSpec_clean_a)

    FullSpec_clean_b = np.log10(FullSpectra_b) - np.log10(aux_b)
    FullSpec_clean_b = np.where(FullSpec_clean_b<=0.0, 0, FullSpec_clean_b)

    # Limpieza de bandas *********************************************
    #FullSpec_clean_a = delete_bands(6,FullSpec_clean_a) #ancho = 6 Hz
    #FullSpec_clean_b = delete_bands(6,FullSpec_clean_b)

    #### Uso del metodo estadìstico percentil 98 o 97
    auxperc_a = np.percentile(FullSpec_clean_a, q=percentil_a, axis=(0,1))
    auxperc_b = np.percentile(FullSpec_clean_b, q=percentil_b, axis=(0,1))

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
    try:
        New_Full_Spectra_a = DBSCAN_algorythm(FullSpec_clean_a,FullSpectra_a,vecinos_a)
    except ValueError:
        print("Datos de matrix: NAN")
        pass
    
    try:
        New_Full_Spectra_b = DBSCAN_algorythm(FullSpec_clean_b,FullSpectra_b,vecinos_b)
    except ValueError:
        print("Datos de matrix: NAN")
        continue
    

    min_a = np.min(FullSpectra_a)
    min_b = np.min(FullSpectra_b)

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
        Guardado_reducted(CurrentSpec,path_o,Pow_reduc_a,Pow_reduc_b,Noise_a,Noise_b)

        del Spectra_a
        del Spectra_b

    if clean == 1:
        
        FS_filtrado_A = New_Full_Spectra_a.toarray()
        
        FS_filtrado_A[FS_filtrado_A == 0] = Noise_a   ### Agregado el reemplazo del min_a por  Noise_a
        pw_ch0 = spec_to_freq(FS_filtrado_A)
        ##
           
        FS_filtrado_B = New_Full_Spectra_b.toarray()
        
        FS_filtrado_B[FS_filtrado_B == 0] = Noise_b   ### Agregado el reemplazo del min_b por  Noise_b
        pw_ch1 = spec_to_freq(FS_filtrado_B)

        print("CurrentSpec!",CurrentSpec)
        path_out   = path_o+folder+'/'
        print("PATH_O:",path_out)
        Guardado_same(CurrentSpec, path_out, pw_ch0,pw_ch1,Noise_a,Noise_b,NoiseRGB_a,NoiseRGB_b)


        #Guardado_same(path, pw_ch0,pw_ch1,Noise_a,Noise_b)
        del FS_filtrado_A
        del FS_filtrado_B

    if plot == 1:
        print("** PLOTEO DE COMPARACION**","nc:",nc)

        ploteado(New_Full_Spectra_a,Noise_a,New_Full_Spectra_b,Noise_b,FullSpectra_a,FullSpectra_b,name,graphics_folder)


if deleted == 1:
    print("Borrando")
    delete_folder(path,files)
