#!/usr/bin/env python
#"""
#@author: JeanFranco, Edited: Vanessa V.
#"""
import matplotlib as mpl
mpl.rc('text', usetex = False)
import matplotlib.pyplot as plt
import numpy as np
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
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from scipy.sparse import csr_matrix

#from sklearn.neighbors import kneighbors_graph
#from sklearn import preprocessing

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
    print("EPS CALCULADO CON KNEED:",eps)

    if eps > 10.5:
        eps = 5
        print("EPS editado con KNEED",eps)
    elif eps < 2.5:
        eps= 4
        print("EPS editado con KNEED",eps)
        
    return eps

def Interference(matrix):
    NoiseFloor = np.median(matrix)
    Prom = np.mean(NoiseFloor)
    Interferencia = np.where(NoiseFloor > Prom,NoiseFloor-Prom,0)

 
    
    
    Freqs = np.linspace(-5,5,matrix.shape[1])
    #NewFullSpectra = FullSpectra
    x_min = -ancho/2
    x_max =  ancho/2
    index_min = int(round((x_min-Freqs[0])/(Freqs[1]-Freqs[0])))
    index_max = int(round((x_max-Freqs[0])/(Freqs[1]-Freqs[0])))
    min_value = np.min(matrix)
    NoiseFloor_Ranges = np.median(matrix, axis=0)
    Prom = np.mean(NoiseFloor_Ranges)
    print("Prom",Prom)
    print("NOISE",NoiseFloor_Ranges)

    #for i in range(len(NoiseFloor_Ranges)):
    indices = np.where(NoiseFloor_Ranges> Prom)

    print("INDICES MAYORES:",indices)
    values = 

    #aux = np.reshape(NoiseFloor_Ranges, (1,matrix.shape[1]))
       
    # Ploteo de espectro original
    plt.figure(figsize=(8,8), dpi=100)
    #plt.subplot(1,6,1)

    plt.scatter(Freqs,NoiseFloor_Ranges)
    #plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra), cmap='jet')
    #plt.pcolormesh( np.log10(FullSpectra), cmap='jet')
    plt.xlabel("Mediana")
    plt.ylabel("Alturas")

    plt.colorbar()

def delete_bands(ancho,matrix):
    Freqs = np.linspace(-5,5,matrix.shape[1])
    #NewFullSpectra = FullSpectra
    x_min = -ancho/2
    x_max = ancho/2
    print(Freqs[0])
    index_min = int(round((x_min-Freqs[0])/(Freqs[1]-Freqs[0])))
    index_max = int(round((x_max-Freqs[0])/(Freqs[1]-Freqs[0])))
    #min_value = np.min(FullSpectra)
    print("index_min",index_min,type(index_min),"index_max",index_max,type(index_max))
    NewFullSpectra = np.zeros((matrix.shape[0],matrix.shape[1]))
    NewFullSpectra[:,index_min:index_max] = matrix[:,index_min:index_max]

    return NewFullSpectra

PathToData = '/home/soporte/Reduccion_de_datos/'
BeginTime = datetime.datetime(2022,4,19,0)
PathToData = PathToData+'d2022110/'
ListFileLinks = sorted(os.listdir(PathToData))
print(ListFileLinks)

NumLinks = len(ListFileLinks)
ArrayMinutes = np.zeros((1,NumLinks)).astype(int)
FileHour, FileMin = 0,6
CurrentLink, Channel = '01',0
LinkDir = PathToData+'sp'+CurrentLink+'_f'+str(Channel)+'/'
print(LinkDir)

FileTime= str(FileHour).zfill(2)+':'+str(FileMin).zfill(2)
ListaSpecFiles = sorted(os.listdir(LinkDir))
print("FileTime:",FileTime)

CurrentSpecFiles = ListaSpecFiles[0]
TimeLabel = np.int(CurrentSpecFiles[5:17])
print("Tiempo:",str(datetime.datetime.fromtimestamp(TimeLabel)))
Minute = int((datetime.datetime.fromtimestamp(TimeLabel)-BeginTime).total_seconds()/60)
### HORA:MINUTO ###
tiempo = str(datetime.datetime.fromtimestamp(TimeLabel))
TimeLabel = str(datetime.datetime.fromtimestamp(TimeLabel))[-8:-3]

if TimeLabel == FileTime:
    print(LinkDir+CurrentSpecFiles)
    #ruta_file = "/media/soporte/PROCDATA/CAMPAIGN/d2022241/sp01_f0/spec-001661763742.hdf5"
    ruta_file = "/media/soporte/PROCDATA/CAMPAIGN/d2022241/sp21_f0/spec-001661751443.hdf5"
    #ruta_file = "/home/soporte/Spectros/CAMPAIGN/d2022241/sp01_f0/spec-001661755751.hdf5"
    #ruta_file = "/home/soporte/Spectros/CAMPAIGN/d2022241/sp01_f0/spec-001661779989.hdf5"
    #f = h5py.File(LinkDir+CurrentSpecFiles,'r')
    f = h5py.File(ruta_file,'r')
    print(f.keys())
    FullSpectra = np.fft.fftshift(np.array(f['pw0_C2']),axes=0).T
    #FullSpectra = np.fft.fftshift(np.array(f['pw0_C2']),axes=0).T
    print("0:",FullSpectra.shape[0],"1:",FullSpectra.shape[1])
    #print(FullSpectra[0])
    f.close()

#Freqs = np.linspace(-5,5,FullSpectra)
#Cantidad de elementos en ejes
#print("FullSpectra original:",FullSpectra[50])
Freqs = np.linspace(-5,5,FullSpectra.shape[1])
Ranges = np.linspace(0,1500,FullSpectra.shape[0])

NoiseFloor_Ranges = np.median(FullSpectra, axis=0) #Matriz que contiene las medianas
Filt_med(5,FullSpectra)
min_value = np.min(FullSpectra)
#print(min)
#print(Freqs2d)
#print("Noise-Floor-Ranges",NoiseFloor_Ranges[50])
#print("Forma de matriz Noise (mediana)",NoiseFloor_Ranges.shape)
aux = np.reshape(NoiseFloor_Ranges, (1,FullSpectra.shape[1]))
#print("FullSpectra-Noise = AUX",aux[0][50])
print("AUX",aux.shape)

#Filt_med(3.3,Freqs,FullSpectra)
# Ploteo de espectro original
plt.figure(figsize=(20,5), dpi=100)
plt.subplot(1,5,1)
plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra), cmap='jet')
#plt.pcolormesh( np.log10(FullSpectra), cmap='jet')
plt.xlabel("Frecuencia")
plt.ylabel("Alturas")
plt.title("%s"%(tiempo))
#plt.xlim(-2,2)
#plt.ylim(0, 500)
plt.colorbar()
#plt.subplot(1,3,2)
#

'''  
plt.subplot(1,6,2)
plt.pcolormesh(Freqs, Ranges, np.log10(NEW_FullSpectra), cmap='jet')
#plt.pcolormesh( np.log10(FullSpectra), cmap='jet')
plt.xlabel("Frecuencia")
plt.ylabel("Alturas")
plt.title("%s,Inter removed"%(tiempo))
plt.colorbar()
# Ploteo de espectro eliminando medianas
'''
FullSpectra_med = np.log10(FullSpectra)-np.log10(aux)
FullSpectra_med = np.where(FullSpectra_med<=0.0, 0, FullSpectra_med)

#FullSpectra_med = np.where(FullSpectra_med<=0.0, min, FullSpectra_med)
print("")
#print("FullSpectra_med: ",FullSpectra_med[50])
plt.subplot(1,5,2)
plt.pcolormesh(Freqs, Ranges, FullSpectra_med, cmap='jet')
plt.xlabel("Frecuencia")
plt.ylabel("Alturas")
plt.title("%s - %s"%(tiempo,"Filt med"))
plt.xlim(-5,5)
plt.ylim(0, 1500)

plt.colorbar()

####### Metodo del percentil ######
'''
plt.subplot(1,4,3)
auxperc_0 = np.percentile(FullSpectra, q=99, axis=(0,1))
plt.pcolormesh(Freqs2d, Ranges2d,(auxperc_0<FullSpectra)*1, cmap='jet')
plt.colorbar()
'''
plt.subplot(1,5,3)
FullSpectra_med = delete_bands(5,FullSpectra_med)
auxperc_1 = np.percentile(FullSpectra_med, q=99, axis=(0,1))
#print("AUX_PERC: ", auxperc_1)

### Grafica de data en forma booleana ###
## Solo se esta graficando los valores mayores al 99 percentil en forma de 1 ##
plt.pcolormesh(Freqs, Ranges, (auxperc_1<=FullSpectra_med)*1, cmap='jet')

plt.colorbar()
plt.title("%s Grafica booleana"%(tiempo))
''' 
############### Calcular los valores del resto ######################
resto = FullSpectra_med-auxperc_1
resto = np.where(resto <=0.0,0,resto)
#print("RESTO",resto[50])
'''
print("Figura 1")
print("close Fig 1.")

Pot = (auxperc_1<=FullSpectra_med)*1

a = (Pot >0)*1
#print("A:",Pot[50])
#Importante obtener el minimo valor de cada espectro
#FullSpectra = np.log10(FullSpectra)

for i in range(1000):
    for j in range(FullSpectra.shape[1]):
        if Pot[i][j] == 0:
            FullSpectra_med[i][j] = 0
            ### Comentar 
            FullSpectra[i][j] = min_value 

#for i in range(1000):
#    for j in range(100):
#        if Pot[i][j] != 0:'
#            FullSpectra_med[i][j] += FullSpectra_med[i][j] + np.log10(aux[0][j])
spectra =  np.fft.ifftshift(FullSpectra)
original = np.fft.fftshift(spectra)
plt.subplot(1,5,4)
plt.pcolormesh(Freqs, Ranges, FullSpectra_med, cmap='jet')
plt.title('%s - Con percentil'%(tiempo))
plt.xlim(-2.5,2.5)
#plt.ylim(0, 1500)
plt.xlabel("Frecuencia")
plt.ylabel("Alturas")

plt.colorbar()
plt.subplot(1,5,5)
#plt.pcolormesh(Freqs2d, Ranges2d, np.log10(FullSpectra), cmap='jet')
plt.pcolormesh(Freqs, Ranges, np.log10(original), cmap='jet')
plt.title('Original con percentil')
#plt.xlim(-2.5,2.5)
plt.ylim(0, 1500)
#plt.xlim(-2,2)
#plt.ylim(0, 500)
plt.colorbar()

#plt.show()

sparse_matrix = sparse.csr_matrix(FullSpectra_med)
#print("The sparse matrix is:")
#print(sparse_matrix[1])
comp = sparse_matrix.tocoo()
Frecuencia=comp.col
Rango = comp.row

#print("Frecuencia:",Frecuencia)
#print("Alturas:",Rango)
plt.figure(figsize=(12,4))
plt.subplot(1,4,1)
plt.scatter(Frecuencia,Rango,s=10,cmap='jet')
plt.title('Espectro con filtro percentil')

#plt.show()

# Creamos un DataFrame para guardar los valores de Frecuencia y Rango
data=pd.DataFrame()
data['Frecuencia']=Frecuencia
data['Rango']=Rango
#data.head()

# Ejecutamos DBSCAN
#dbscan = DBSCAN(eps=0.01, min_samples=10, metric = "euclidean").fit(data)
eps=10
clusters = DBSCAN(eps=eps, min_samples=5).fit_predict(data)
#clusters = dbscan.fit_predict(data
#print("CLUSTERS",clusters)
df_values = data.values
plt.subplot(1,4,2)
#plt.scatter(df_values[:, 0], df_values[:, 1], c=clusters,s=100, cmap="plasma")
plt.scatter(df_values[:, 0], df_values[:, 1], c=clusters,s=10)
plt.title('APLICACIÓN DBSCAN- EPS:%s'%(eps))
plt.box("False")

## DBSCAN 2
eps=8
clusters = DBSCAN(eps=eps, min_samples=10, metric = "euclidean").fit_predict(data)
#clusters = dbscan.fit_predict(data
#print("CLUSTERS",clusters)
df_values = data.values
plt.subplot(1,4,3)
#plt.scatter(df_values[:, 0], df_values[:, 1], c=clusters,s=100, cmap="plasma")
plt.scatter(df_values[:, 0], df_values[:, 1], c=clusters,s=10)
plt.title('APLICACIÓN DBSCAN- EPS:%s'%(eps))
plt.box("False")


## DBSCAN 3
eps=5
clusters = DBSCAN(eps=eps, min_samples=10, metric = "euclidean",).fit_predict(data)
#clusters = dbscan.fit_predict(data
#print("CLUSTERS",clusters)
X = df_values = data.values
plt.subplot(1,4,4)
#plt.scatter(df_values[:, 0], df_values[:, 1], c=clusters,s=100, cmap="plasma")
plt.scatter(df_values[:, 0], df_values[:, 1], c=clusters,s=10)
plt.title('APLICACIÓN DBSCAN- EPS:%s'%(eps))
plt.box("False")

#plt.show()

'''  
# Parametrización de DBSCAN
estimator = PCA (n_components = 2)

X_pca = estimator.fit_transform(data)
print("Estimator: ",X_pca)
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
print("DIST: ",dist)
matsim = dist.pairwise(X_pca)
minPts  = 5



 # Fijamos el parámetro minPts
A = kneighbors_graph(X_pca, minPts, include_self=False)
Ar = A.toarray()
print("AR:",Ar.shape,Ar)
seq = []
for i,s in enumerate(X_pca):
    for j in range(len(X_pca)):
        if Ar[i][j] != 0:
            seq.append(matsim[i][j])
seq.sort()
#print("SEQ:",seq)
'''

vecinos = 10
neigh = NearestNeighbors(n_neighbors = vecinos)
nbrs = neigh.fit(X)
distancias,indices = nbrs.kneighbors(X)

distancias = np.sort(distancias,axis=0)
distancias = distancias[:,vecinos -1]
distancias = sorted(distancias)
#print(distancias)
# Aplicamos el método de la curva L
#i = np.arange(len(seq))
i = np.arange(len(distancias))
knee = KneeLocator(i, distancias, curve='convex', direction='increasing', interp_method='polynomial',online= True,S=1)
eps=distancias[knee.knee]
print("EPS ORIGINAL CALCULADO CON KNEED:",eps)
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")
print('El valor de eps adecuado es: ', eps)

eps = epsilon(data, vecinos = 10)
# Ejecutamos DBSCAN nuevamente
dbscan = DBSCAN(eps=eps, min_samples = vecinos, metric = "euclidean").fit(X) #"X o data"
cluster = dbscan.labels_
#clusters = dbscan.fit_predict(data)
df_values = data.values
plt.figure(figsize=(12,6))
plt.subplot(1,4,1)
plt.scatter(df_values[:, 0], df_values[:, 1], c=clusters, cmap="plasma")
plt.title('DBSCAN-AJUSTE de EPS: %s'%(eps))

# Creamos un nuevo DataFrame con una columna adicional que indentifica los clusters
copy = pd.DataFrame()
copy['Frecuencia']=data['Frecuencia'].values
copy['Rango']=data['Rango'].values
copy['label'] =  cluster
#cantidadGrupo =  pd.DataFrame()
#cantidadGrupo['cantidad']=copy.groupby('label').size()

#Eliminamos los puntos marcados como outliers y representamos gráficamente.
copy = copy.drop(copy[copy['label'] == -1].index)

# Dataframe "limpio"
f1 = copy['Frecuencia'].values
f2 = copy['Rango'].values

# Ploteo scatter del espectro filtrado modificado
plt.subplot(1,4,2)
plt.scatter(f1, f2,s=2)

#plt.show()

New_Full_Spectra = []

for k in range(len(f1)):

    New_Full_Spectra.append(FullSpectra[f2[k]][f1[k]])
#### **** **** FIN *** *** ####
New_Full_Spectra = np.array(New_Full_Spectra)
#### **** **** FIN *** *** ####

New_Full_Spectra = csr_matrix((New_Full_Spectra,(f2,f1)),shape=(1000,FullSpectra.shape[1]))
#print("MATRIX OBTENIDA",New_Full_Spectra)

FS_filtrado = New_Full_Spectra.toarray()
FS_filtrado[FS_filtrado == 0] = min_value

plt.subplot(1,4,3)
plt.pcolormesh(Freqs, Ranges,np.log10(FS_filtrado),cmap='jet')
plt.xlabel("Frecuencia")
plt.ylabel("Alturas")
plt.title("%s"%(tiempo))
plt.colorbar()

pw_ch0 = np.fft.ifftshift(FS_filtrado.T)
#with h5py.File('%s'%(LinkDir+CurrentSpecFiles),'r+') as hf:
#    del hf['pw0_C0']

#    hf.create_dataset('pw0_C0',data = pw_ch0)

#        f['pw0_C%s'%(0)]  =   pw_ch0
        #f['pw1_C%s'%(code)]  =   pw_ch1

print("****MODIFICADO ESPECTRO****")
#### * * *  * * * * * * *  MATRIX PARA ALMACENAR EN HDF5

ncomp = New_Full_Spectra.tocoo()
#print("Longitud",len(ncomp.col))
one_frecuencia = []
one_rango = []

Pow_reduc_a = np.zeros((len(ncomp.col),3))
Pow_reduc_a[:,0],Pow_reduc_a[:,1],Pow_reduc_a[:,2] = ncomp.col, ncomp.row, ncomp.data

 
#print("COMPONENTES:",ncomp)
#print("DATA GUARDADA:",Pow_reduc_a)
#### * **  * * * * * * *


##### ****** METODO DE LECTURA  ****** #####
Pow = Pow_reduc_a[:,2]
f2 = Pow_reduc_a[:,1]
f1 = Pow_reduc_a[:,0]
print("Tipo F2:",type(f2),f1.shape,f2.shape)


matrix_read = csr_matrix((Pow,(f2,f1)),shape= (1000,FullSpectra.shape[1]))
#print("ARTE:",New_Full_Spectra == matrix_read)
matrix_read = matrix_read.toarray()
matrix_read[matrix_read == 0] = min_value

#print("Matrix leida:", matrix_read)
#print("Longitud",len(f1))


Matrix_leida = New_Full_Spectra.toarray()
Matrix_leida [Matrix_leida  == 0] = min_value
#print(matrix_read == Matrix_leida )

#Lectura
print("Min:",min_value)
#FS_filtrado[FS_filtrado == 0] = min_value
#print("FIN2",FS_filtrado[152])
plt.subplot(1,4,4)
plt.pcolormesh(Freqs, Ranges,np.log10(Matrix_leida),cmap='jet')
#plt.pcolormesh(Freqs, Ranges,np.log10(np.fft.fftshift(New_Full_Spectra)),cmap='jet')
plt.colorbar()
plt.show()

'''  
cop = pd.DataFrame()
cop['Potencia'] = comp.data
cop['label'] = clusters;
cop = cop.drop(cop[cop['label'] == -1].index)

f3 = cop['Potencia'].values

'''

def Lectura_specreducted():
    pass

ruta_dir = '/media/soporte/PROCDATA/d2022238/sp01_f0.hdf5'
f = h5py.File(ruta_dir,'r')
channel=['0','1']
code = '0'
for chan in channel:
    for i in f.keys():
        power_a = np.array(f[i+'/pw'+chan+'_C'+code])
        #power_b = 
        print(i)
        print(power_a)




