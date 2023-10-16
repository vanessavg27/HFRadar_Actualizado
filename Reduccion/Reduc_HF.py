#!/usr/bin/env python3.8
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime
import os,sys
import h5py
from scipy import sparse
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from scipy.sparse import csr_matrix
import argparse
import glob
import math,shutil
import _noise
from colorama import Fore, Back, Style

class Reduccion():
    def __init__(self,op):
        self.path            = op.path_lectura
        self.path_o          = op.path_out
        self.graphics_folder = op.graphics_folder
        self.campaign        = op.c_campaign
        self.freqs	         = op.f_freq 
        self.code	         = int(op.code_seleccionado)
        self.Days	         = op.date_seleccionado
        self.lo		         = str(op.lo_seleccionado)
        self.reducted        = int(op.reducted)
        self.hild_sk         = op.hild_sk
        self.remove_dc       = int(op.remove_dc)
        self.start_h         = str(op.start_h)
        self.end_h           = str(op.end_h)
        ################ REDUCED FLAG
        self.reducted        = int(op.reducted)
        ################ BORRAR DATOS FLAG
        self.deleted         = int(op.delete)
        ################ PLOT DATOS FLAG
        self.plot            = op.plot
        self.nrange          = 1000
        self.old             = op.old
        ################ DATAFRAME CON COORDENADAS
        

        if self.campaign == 1:
            self.nc        = 600
            self.path      = self.path + "CAMPAIGN/"
            self.path_o    = self.path_o + "CAMPAIGN/"
            self.mode      = 'Campaign'
            self.int_incoh = 1
            self.pad       = 3
            self.separacion= 40
        else:
            self.nc        = 100
            self.path      = self.path
            self.path_o    = self.path_o
            self.mode      = 'Normal'
            self.int_incoh = 6
            self.pad       = 4
            self.separacion= 10

        if self.freqs < 3:
            self.ngraph      = 0
        else:
            self.ngraph      = 1

        self.rx = {'11':"JRO-A",'12':"JRO-B",'21':"HYO-A",'22':"HYO-B",'31':"MALA",'41':"MERCED",'51':"BARRANCA",'61':"LA OROYA"}
        self.tx = {'0':'Ancon','1':'Sicaya','2':'Ica'}

        print("*** Iniciando PROC - FILTRADO Y REDUCCION***")
        print("      Estacion Rx:        ",self.rx[self.lo])
        print("      Estacion Tx:        ",self.tx[str(self.code)])
        print("      Modo de Operación:  ",self.mode)
        print("      Fecha:              ",self.Days)        
        print("      Contorno de relleno:",self.pad)

        if self.old == 1:
            print("      Misma banda de frecuencia")
            print("      Porcentaje ch0:     ",self.samefreqs(self.lo,self.mode,str(self.freqs),'ch0',0,self.code))
            print("      Porcentaje ch1:     ",self.samefreqs(self.lo,self.mode,str(self.freqs),'ch1',0,self.code))
        else:
            print("Multifrecuencia")
            print("      Porcentaje ch0:     ",self.multifreqs(self.lo,self.mode,str(self.freqs),'ch0',0,self.code))
            print("      Porcentaje ch1:     ",self.multifreqs(self.lo,self.mode,str(self.freqs),'ch1',0,self.code))

        days   = datetime.datetime.strptime(self.Days, "%Y/%m/%d")
        day    =  days.strftime("%Y%j")
        folder = "sp%s1_f%s"%(self.code,self.ngraph)
        path_o   = self.path_o+"d"+day+'/'+folder
        if os.path.exists('%s.hdf5'%(path_o)) and self.reducted == 1:
            print("    Elimino archivo existente: %s.hdf5"%(path_o))
            os.remove('%s.hdf5'%(path_o))
        
        ##EJECUCION 
        self.clean_run()

    def h5_list(self,code,type_freq):
        days                 = datetime.datetime.strptime(self.Days, "%Y/%m/%d")
        day                  = days.strftime("%Y%j")

        folder               = "sp%s1_f%s"%(code,type_freq)
        path                 = self.path+"d"+day+'/'+folder
        self.graphics_folder = self.graphics_folder+"d"+day+'/'+folder+"/"
        self.path_o          = self.path_o+"d"+day+'/'

        files                = glob.glob(path+"/spec-*.hdf5")
        files.sort()
        print("      **",folder)
        
        archivos = [x for x in files if (datetime.datetime.fromtimestamp(int(x.split("/")[-1][5:-5])).strftime("%H:%M:%S") >= self.start_h and datetime.datetime.fromtimestamp(int(x.split("/")[-1][5:-5])).strftime("%H:%M:%S") <= self.end_h)]

        if archivos == [] or len(archivos) == 1:
            print(Fore.RED+"No hay archivos en ese rango de horas")
            return 0
        
        return archivos

    def clean_run(self):

        for CurrentSpec in self.h5_list(self.code,self.ngraph):
            print("  **Archivo:",CurrentSpec)
            try:
                ch0 = self.read_h5(CurrentSpec,0)
            except:
                continue
            try:
                ch1 = self.read_h5(CurrentSpec,1)
            except:
                continue 
            
            tiempo   = str(datetime.datetime.fromtimestamp(int(self.name)))
            print("      **TIME:",tiempo, end=" ")
            FullSpectra_a = np.fft.fftshift(ch0,axes=0).T
            FullSpectra_b = np.fft.fftshift(ch1,axes=0).T
            
            if self.remove_dc == 1:
                FullSpectra_a = self.removeDC(FullSpectra_a.T,self.nc,self.nrange, mode=2).T
                FullSpectra_b = self.removeDC(FullSpectra_b.T,self.nc,self.nrange, mode=2).T

            ch0=self.light_fil(self.spec_to_freq(FullSpectra_a).T).T
            ch1=self.light_fil(self.spec_to_freq(FullSpectra_b).T).T

            FullSpectra_a = np.fft.fftshift(ch0,axes=0).T
            FullSpectra_b = np.fft.fftshift(ch1,axes=0).T
            #Solo afecta a modo campaña
            ruido_a    = np.delete(FullSpectra_a, range(120,500), axis=0)
            ruido_b    = np.delete(FullSpectra_b, range(120,500), axis=0)

            NoiseFloor_a = np.median( ruido_a, axis=0)
            NoiseFloor_b = np.median( ruido_b, axis=0)

            if self.nc == 100:
                #Normal mode
                NoiseFloor_a = np.median(FullSpectra_a, axis=0)
                NoiseFloor_b = np.median(FullSpectra_b, axis=0)
                
                if self.old == 1:
                    #Same frequencies
                    percentil_a  = self.samefreqs(self.lo,self.mode,str(self.freqs),'ch0',0,self.code)
                    percentil_b  = self.samefreqs(self.lo,self.mode,str(self.freqs),'ch1',0,self.code)                
                    vecinos_a    = self.samefreqs(self.lo,self.mode,str(self.freqs),'ch0',1,self.code)
                    vecinos_b    = self.samefreqs(self.lo,self.mode,str(self.freqs),'ch1',1,self.code)
                else:
                    #Multifrequencies
                    percentil_a  = self.multifreqs(self.lo,self.mode,str(self.freqs),'ch0',0,self.code)
                    percentil_b  = self.multifreqs(self.lo,self.mode,str(self.freqs),'ch1',0,self.code)
                    vecinos_a    = self.multifreqs(self.lo,self.mode,str(self.freqs),'ch0',1,self.code)
                    vecinos_b    = self.multifreqs(self.lo,self.mode,str(self.freqs),'ch1',1,self.code)

            else:
                #Campaign Mode
                if self.old == 1:
                    #Same frequencies
                    percentil_a  = self.samefreqs(self.lo,self.mode,str(self.freqs),'ch0',0,self.code)
                    percentil_b  = self.samefreqs(self.lo,self.mode,str(self.freqs),'ch1',0,self.code)               
                    vecinos_a    = self.samefreqs(self.lo,self.mode,str(self.freqs),'ch0',1,self.code)
                    vecinos_b    = self.samefreqs(self.lo,self.mode,str(self.freqs),'ch1',1,self.code)
                else:
                    #Multifrequencies
                    percentil_a  = self.multifreqs(self.lo,self.mode,str(self.freqs),'ch1',0,self.code)
                    percentil_b  = self.multifreqs(self.lo,self.mode,str(self.freqs),'ch1',0,self.code)
                    vecinos_a    = self.multifreqs(self.lo,self.mode,str(self.freqs),'ch0',1,self.code)
                    vecinos_b    = self.multifreqs(self.lo,self.mode,str(self.freqs),'ch1',1,self.code)

            
            aux_a =  np.reshape(NoiseFloor_a, (1,self.nc))
            aux_b =  np.reshape(NoiseFloor_b, (1,self.nc))

            FullSpec_clean_a = np.log10(FullSpectra_a) - np.log10(aux_a)
            FullSpec_clean_a = np.where(FullSpec_clean_a<=0.0, 0, FullSpec_clean_a)

            FullSpec_clean_b = np.log10(FullSpectra_b) - np.log10(aux_b)
            FullSpec_clean_b = np.where(FullSpec_clean_b<=0.0, 0, FullSpec_clean_b)

            #### Uso del metodo estadìstico percentil 98 o 97
            auxperc_a = np.percentile(FullSpec_clean_a, q=percentil_a, axis=(0,1))
            auxperc_b = np.percentile(FullSpec_clean_b, q=percentil_b, axis=(0,1))

            Pot_a = (auxperc_a<=FullSpec_clean_a)*1
            Pot_b = (auxperc_b<=FullSpec_clean_b)*1

            for i in range(self.nrange):
                for j in range(self.nc):
                    if Pot_a[i][j] == 0:
                        
                        FullSpec_clean_a[i][j] = 0

                    if Pot_b[i][j] == 0:
                        
                        FullSpec_clean_b[i][j] = 0

            try:
                New_Full_Spectra_a = self.DBSCAN_algorythm(FullSpec_clean_a,FullSpectra_a,vecinos_a)
                #self.df_out.plot(x='Frecuencia', y='Rango',kind='scatter',title=" Padding %s"%(self.pad))
                #plt.show()
            except ValueError:
                print("      Datos de matrix: NAN")
                pass
#           ######################## PROGRAMAR NUEVO RUIDO ###############
                        
            Noisegeneral_a = self.Ruido_oficial(ch0,self.nrange)
#           ##############################################################
            try:
                New_Full_Spectra_b = self.DBSCAN_algorythm(FullSpec_clean_b,FullSpectra_b,vecinos_b)
                

            except ValueError:
                print("      Datos de matrix: NAN")
                continue

#           ######################## PROGRAMAR NUEVO RUIDO ###############
            
            Noisegeneral_b = self.Ruido_oficial(ch0,self.nrange)
#           ##############################################################

            if self.reducted == 1:

                Spectra_a   =New_Full_Spectra_a.toarray()
                #Spectra_a = np.fft.ifftshift(Spectra_a.T,axes=0)
                Spectra_a   = self.spec_to_freq(Spectra_a)
                ncomp_a     = csr_matrix(Spectra_a).tocoo()
                #ncomp_a = Spectra_a.tocoo()
                Pow_reduc_a = np.zeros((len(ncomp_a.col),3))
                Pow_reduc_a[:,0],Pow_reduc_a[:,1],Pow_reduc_a[:,2] = ncomp_a.col, ncomp_a.row, ncomp_a.data

                Spectra_b   = New_Full_Spectra_b.toarray()
                #Spectra_b = np.fft.ifftshift(Spectra_b.T,axes=0)
                Spectra_b   = self.spec_to_freq(Spectra_b)
                ncomp_b     = csr_matrix(Spectra_b).tocoo()
                Pow_reduc_b = np.zeros((len(ncomp_b.col),3))
                Pow_reduc_b[:,0],Pow_reduc_b[:,1],Pow_reduc_b[:,2] = ncomp_b.col, ncomp_b.row, ncomp_b.data

                #Guardado_reducted(CurrentSpec,Pow_reduc_a,Pow_reduc_b,Noise_a,Noise_b, min_a, min_b)
                self.Guardado_reducted(CurrentSpec,self.path_o,Pow_reduc_a,Pow_reduc_b,Noisegeneral_a,Noisegeneral_b)
                
                del Spectra_a
                del Spectra_b

            if self.plot == 1:
                print("** PLOTEO DE COMPARACION**","nc:",self.nc)

                self.ploteado(New_Full_Spectra_a,Noisegeneral_a,New_Full_Spectra_b,Noisegeneral_b,FullSpectra_a,FullSpectra_b,self.name,self.graphics_folder)
    
    def multifreqs(self,lo,mode,freqs,ch,variable,code):
        
        location = {'11':{'Campaign':{'2.72':{'ch0':[[86,86,86],[38,38,37]],'ch1':[[86,86,86],[38,38,37]]},
                                      '3.64':{'ch0':[[86,86,87],[38,38,40]],'ch1':[[88,87,88],[38,38,38]]}},
                            'Normal':{'2.72':{'ch0':[[92.5,92,92],[23,22,15]],'ch1':[[94,92,92],[24,20,21]]},
                                      '3.64':{'ch0':[[93,93,92],[23,20,20]],'ch1':[[93,94,93],[22,20,21]]}}}, #JROA Good

                    '12':{'Campaign':{'2.72':{'ch0':[[88,94,90],[38,38,36]],'ch1':[[86,92,90],[38,38,37]]},
                                      '3.64':{'ch0':[[89,90,88],[38,40,37]],'ch1':[[88,86,86],[39,42,37]]}},
                            'Normal':{'2.72':{'ch0':[[92,92,92],[22,21,18]],'ch1':[[93,93,93],[22,22,18]]},
                                      '3.64':{'ch0':[[94,93,92],[24,23,22]],'ch1':[[93,92,92],[24,23,22]]}}}, #JROB Good

                    '21':{'Campaign':{'2.72':{'ch0':[[86,86,86],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]},
                                      '3.64':{'ch0':[[86,86,86],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]}},
                            'Normal':{'2.72':{'ch0':[[91,91,91],[30,30,30]],'ch1':[[95,95,95],[18,18,18]]},
                                      '3.64':{'ch0':[[91,91,91],[30,30,30]],'ch1':[[95,95,95],[18,18,18]]}}}, #HYOA Good
            
                    '22':{'Campaign':{'2.72':{'ch0':[[86,86,86],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]},
                                      '3.64':{'ch0':[[86,86,86],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]}},
                            'Normal':{'2.72':{'ch0':[[92,91,91],[28,29,30]],'ch1':[[95,95,94],[18,19,20]]},
                                      '3.64':{'ch0':[[92,91,91],[28,29,30]],'ch1':[[95,95,94],[18,19,20]]}}}, #HYOB
            
                    '31':{'Campaign':{'2.72':{'ch0':[[87,88,88],[38,38,38]],'ch1':[[89,92,86],[36,30,38]]},
                                      '3.64':{'ch0':[[88,88,88],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]}},
                            'Normal':{'2.72':{'ch0':[[96,95,96],[15,18,15]],'ch1':[[95,96,97],[20,15,15]]},
                                      '3.64':{'ch0':[[92,90,94],[27,30,20]],'ch1':[[96,98,96],[14,14,18]]}}}, #Mala Good
            
            '41':{'Campaign':{'2.72':{'ch0':[[96,96,96],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]},   #Good
                              '3.64':{'ch0':[[88,88,88],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]}},  
                    'Normal':{'2.72':{'ch0':[[95,97,97],[20,20,16]],'ch1':[[95,94,94],[19,22,18]]},
                              '3.64':{'ch0':[[92,92,92],[20,20,20]],'ch1':[[94,94,94],[20,20,20]]}}}, #Merced

            '51':{'Campaign':{'2.72':{'ch0':[[88,88,87],[35,34,36]],'ch1':[[86,89,86],[38,35,38]]},            
                              '3.64':{'ch0':[[86,84,87],[35,30,36]],'ch1':[[86,85,86],[38,37,38]]}},
                    'Normal':{'2.72':{'ch0':[[88,90,88],[22,28,27]],'ch1':[[92,92,92],[20,22,20]]},   
                              '3.64':{'ch0':[[88,90,88],[22,28,27]],'ch1':[[92,92,92],[20,22,20]]}}}, #Barranca Good

            '61':{'Campaign':{'2.72':{'ch0':[[86,87,82],[42,39,50]],'ch1':[[86,86,86],[40,40,40]]},
                              '3.64':{'ch0':[[87,86,88],[37,35,35]],'ch1':[[86,86,86],[38,38,38]]}},
                    'Normal':{'2.72':{'ch0':[[87,92,87],[37,32,40]],'ch1':[[95,94,90],[18,22,37]]}, 
                              '3.64':{'ch0':[[87,92,89],[37,24,30]],'ch1':[[93,95,92],[24,22,28]]}}}  #Oroya  Good                                     
            }
        return location[lo][mode]["%s"%(freqs)]["%s"%(ch)][variable][code]
        
    def samefreqs(self,lo,mode,freqs,ch,variable,code):
        location = {'11':{'Campaign':{'2.72':{'ch0':[[86,86,86],[38,38,37]],'ch1':[[86,86,86],[38,38,37]]},
                                      '3.64':{'ch0':[[89,87,88],[41,38,40]],'ch1':[[89,86,88],[42,38,40]]}},
                            'Normal':{'2.72':{'ch0':[[94,93,92],[21,18,20]],'ch1':[[93,93,93],[22,20,18]]},
                                      '3.64':{'ch0':[[92,92,92],[21,18,20]],'ch1':[[93,93,93],[20,21,20]]}}}, #JROA Good

            '12':{'Campaign':{'2.72':{'ch0':[[88,90,90],[38,40,36]],'ch1':[[86,90,90],[38,38,37]]},
                              '3.64':{'ch0':[[89,90,88],[38,35,37]],'ch1':[[88,86,86],[39,38,37]]}},
                    'Normal':{'2.72':{'ch0':[[94,93,93],[20,18,19]],'ch1':[[93,93,93],[20,16,16]]},
                              '3.64':{'ch0':[[92,92,92],[15,16,15]],'ch1':[[93,93,93],[16,16,16]]}}}, #JROB Good

            '21':{'Campaign':{'2.72':{'ch0':[[86,86,86],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]},
                              '3.64':{'ch0':[[86,86,86],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]}},
                    'Normal':{'2.72':{'ch0':[[91,91,91],[30,30,30]],'ch1':[[95,95,95],[18,18,18]]},
                              '3.64':{'ch0':[[91,91,91],[30,30,30]],'ch1':[[95,95,95],[18,18,18]]}}}, #HYOA Good
            
            '22':{'Campaign':{'2.72':{'ch0':[[86,86,86],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]},
                              '3.64':{'ch0':[[86,86,86],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]}},
                    'Normal':{'2.72':{'ch0':[[92,91,91],[28,29,30]],'ch1':[[95,95,94],[18,19,20]]},
                              '3.64':{'ch0':[[92,91,91],[28,29,30]],'ch1':[[95,95,94],[18,19,20]]}}}, #HYOB
            
            '31':{'Campaign':{'2.72':{'ch0':[[87,88,88],[38,38,38]],'ch1':[[89,92,86],[36,30,38]]},
                              '3.64':{'ch0':[[88,88,88],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]}},
                    'Normal':{'2.72':{'ch0':[[96,95,96],[15,18,15]],'ch1':[[95,96,97],[20,15,15]]},
                              '3.64':{'ch0':[[92,90,94],[27,30,20]],'ch1':[[96,98,96],[14,14,18]]}}}, #Mala Good
            
            '41':{'Campaign':{'2.72':{'ch0':[[96,96,96],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]},   #Good
                              '3.64':{'ch0':[[88,88,88],[38,38,38]],'ch1':[[86,86,86],[38,38,38]]}},  
                    'Normal':{'2.72':{'ch0':[[95,97,97],[20,20,16]],'ch1':[[95,94,94],[19,22,18]]},
                              '3.64':{'ch0':[[92,92,92],[20,20,20]],'ch1':[[94,94,94],[20,20,20]]}}}, #Merced

            '51':{'Campaign':{'2.72':{'ch0':[[88,88,87],[35,34,36]],'ch1':[[86,89,86],[38,35,38]]},            
                              '3.64':{'ch0':[[86,84,87],[35,30,36]],'ch1':[[86,85,86],[38,37,38]]}},
                    'Normal':{'2.72':{'ch0':[[92,90,88],[26,22,27]],'ch1':[[94,93,93],[22,22,22]]},   
                              '3.64':{'ch0':[[88,90,88],[22,28,27]],'ch1':[[92,92,92],[20,22,20]]}}}, #Barranca Good

            '61':{'Campaign':{'2.72':{'ch0':[[86,87,82],[42,39,50]],'ch1':[[86,86,86],[40,40,40]]},
                              '3.64':{'ch0':[[87,86,88],[37,35,35]],'ch1':[[86,86,86],[38,38,38]]}},
                    'Normal':{'2.72':{'ch0':[[87,92,87],[37,32,40]],'ch1':[[95,94,90],[18,22,37]]}, 
                              '3.64':{'ch0':[[87,92,89],[37,24,30]],'ch1':[[93,95,92],[24,22,28]]}}}  #Oroya  Good                                     
            }
        return location[lo][mode]["%s"%(freqs)]["%s"%(ch)][variable][code]

    def read_h5(self,file,ch):

        try:
            f    = h5py.File(file,'r')
        except OSError:
            print("\nError en el espectro *.hdf5\n")
        
        self.name     = file.split("/")[-1][5:-5]
        
        try:
            channel = np.array(f['pw%s_C%s'%(str(ch),self.code)])
        except KeyError:
            print("      **Espectro vacio \n")
        
        f.close()

        return channel
     
    def light_fil(self,data):
        alturas= data.shape[0]
        freqs= data.shape[1]
        new_pspec = np.zeros((alturas,freqs))
        #print("SIZE!",size)
        for l in range(alturas):
            if l == 0:
                new_pspec[0] = data[0]
            
            elif (l< alturas-2):
                new_pspec[l]= (data[l-1]+data[l]+data[l+1])/5
            else:
                new_pspec[l]=data[l]
        del alturas
        del freqs
        return new_pspec
    
    def Ruido_oficial(self,ch0,nrange):
    
        med_profile = np.median(ch0.T,axis=0)
        med_total   = np.median(ch0.T)

        index_med   = [index for index,value in enumerate(med_profile) if value < med_total]
        newmatrix   = np.zeros((nrange,len(index_med)))

        for i in range(len(index_med)):
            newmatrix[:,i]    = ch0.T[:,index_med[i]]
    
        return np.median(newmatrix)

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

#Funciones de relleno con padding
    def listas(self,v):
        lista   = []
        lista_g = []
        for i in range(len(v)-1):
            #a = i
            a = v[i+1] - v[i]
            if a < self.separacion: #Valor de separacion de ecos en la misma altura
                lista.append(v[i])
            else:
                lista.append(v[i])
                lista_g.append(lista)
                lista=[]

        lista.append(v[-1])
        lista_g.append(lista)
        return lista_g
    
    def relleno(self,lista,pad,index_min,index_max,h):
    
        if pad != 0:
            min=lista[0]-pad
            max=lista[-1]+pad
            if min < index_min:
                min=int(index_min)
            if max >= index_max:
                max = int(index_max)
        else:
            min=lista[0]
            max=lista[-1]

        lista_2= [i for i in range(min,max+1) if not(i in lista)]  
        lista_2=lista_2+lista
        lista_2= sorted(lista_2)
        #print("Lista2:",lista_2)
        nuevo_registro = {'Rango':[h for x in range(len(lista_2))],'Frecuencia':lista_2,'label':[3 for x in range(len(lista_2))]}        
        df_2           = pd.DataFrame(nuevo_registro)
        return df_2
    
    def padding(self,df,pad,nc):
    #lista_copy = [[v[0],v[1]] for v in lista]
        Freqs = np.linspace(-5,5,nc)
        index_min = int(round((-2.5-Freqs[0])/(Freqs[1]-Freqs[0])))
        index_max = int(round(( 2.5-Freqs[0])/(Freqs[1]-Freqs[0])))
        self.df_out          = pd.DataFrame()
        #DATA DBSCAN")
        ancho_min = int(round((-3-Freqs[0])/(Freqs[1]-Freqs[0])))
        ancho_max = int(round(( 3-Freqs[0])/(Freqs[1]-Freqs[0])))

        for h in range(1000):
            a = df[df['Rango']==h]
            if a.empty:
                continue
            if h > 50:       #Considerar todas las frecuencias en la altura 150*1.5 y 400*1.5 km
                #lista = [v for v in a.sort_values(by=['Frecuencia'],ascending=[True])['Frecuencia'].tolist()]
                lista = df[df['Rango']==h]['Frecuencia'].tolist()
            else:
                #lista = [v for v in a.sort_values(by=['Frecuencia'],ascending=[True])['Frecuencia'].tolist() if (v >=ancho_min and v <= ancho_max)]
                a = df[df['Rango']==h]
                lista = a[(a['Frecuencia']<ancho_max) & (a['Frecuencia'] >ancho_min)]['Frecuencia'].tolist()
                
            if len(lista) == 1 or len(lista) == 0:
                continue

            lista_0 = self.listas(lista)

            for l in lista_0:
                df_2         = self.relleno(l,pad,index_min,index_max,h) #Relleno cada altura
                self.df_out  = self.df_out.append(df_2,ignore_index=True)
                
            del(a)
            del(lista)
            del(df_2)
        #self.df_out.plot(x='Frecuencia', y='Rango',kind='scatter',title="SIGNAL-Padding=%s"%(self.pad))
        #plt.show()
        return self.df_out

    def plot_DataFrame(self,data):
        font = {'family': 'serif','color':  'black','weight': 'bold','size': 25}
        plt.figure(figsize=(9,14))
        plt.scatter(data['Frecuencia'], data['Rango'],c=data['label'])
        plt.title("%s "%("DBScan Algorithm"), fontdict=font)
        plt.xlabel("Frequency Index",fontdict=font)
        plt.ylabel("Range Index",fontdict=font)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.ylim((0,1000))
        plt.show()

    def DBSCAN_algorythm(self,matrix,FullSpectra,vecinos):
        plot = 0 # Hardcore. Activar si desea visualizar los plots de datagrama. plot=1
        #Creacion de data sparse (Solo componentes != 0)
        sparse_matrix      = sparse.csr_matrix(matrix).tocoo()
        Frecuencia         = sparse_matrix.col
        Rango              = sparse_matrix.row
        #Colocacion de datos en DataFrame.
        data               = pd.DataFrame()
        data['Frecuencia'] = Frecuencia
        data['Rango']      = Rango
        X_pow              = data.values
        #Hallar el epsilon de agrupaciones.
        eps                = self.epsilon(X_pow, vecinos = vecinos)
        #Ejecucion de DBSCAN.
        db                 = DBSCAN(eps=eps, min_samples = vecinos, metric= "euclidean").fit(X_pow)
        clusters           = db.labels_
        data['label']      = clusters
        if plot:
            self.plot_DataFrame(data) 
        #Eliminación de etiquetas de ruido.
        data_clean1               = data.drop(data[data['label'] == -1].index)
        if data_clean1.empty:
            data_clean1           = data
        #PRUEBA MARCO
        data_pad              = self.padding(data_clean1,self.pad,self.nc)
        if data_pad.empty:
            data_pad              = data_clean1
        if plot:
            self.plot_DataFrame(data_pad)
        #Dataframe limpio
        try:
            data_clean2         = data_pad.drop(data_pad[data_pad['label'] == -1].index)
        except TypeError:
            data_clean2 = data_clean1

        f1_a               = data_clean2['Frecuencia'].values
        f2_a               = data_clean2['Rango'].values

        New_Full_Spectra = []

        for k in range(len(f1_a)):
    
            New_Full_Spectra.append(FullSpectra[f2_a[k]][f1_a[k]])
    
        New_Full_Spectra = np.array(New_Full_Spectra)
        New_Full_Spectra = csr_matrix((New_Full_Spectra,(f2_a,f1_a)),shape=(1000,self.nc))

        del(data)
        del(data_clean1)
        del(data_pad)
        del(data_clean2)
        return New_Full_Spectra


    def epsilon(self,data, vecinos = 10):

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
            print("      EPS EDITADO CON KNEED:",eps,end=" - ")
        elif eps<4:
            eps= 4.3
            print("      EPS EDITADO CON KNEED:",eps,end=" - ")
        else:
            print("      EPS CALCULADO CON KNEED:",eps)
        return eps
    
    def spec_to_freq(self,spec):
        #detectar si es convexo o concavo
        power = np.fft.ifftshift(spec.T,axes=0)
        return power
    
    def delete_folder(self,path,files):
        #print("Funcion ",path+'.hdf5')
        f   = h5py.File(path+'.hdf5','r')
        filelist = sorted(list(f.keys()))
        f.close()
    
        if filelist[-1] == files[-1].split("/")[-1][:-5]:
            print("....Borrando")
            print("")
            print("    FOLDER ELIMINADO:",path)
            shutil.rmtree("%s"%(path))

    def Guardado_reducted(self,path,path_o,pw_ch0,pw_ch1,noise_a,noise_b):
        folder = path.split("/")[-2]
        #print("\n        PATH GUARDADO: ",path)
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
            pass
            #print("Archivo ya existente",path_o)
        
        with h5py.File('%s.hdf5'%(path_o+folder),'a') as f:
        #with h5py.File('d%s.hdf5'%(day),'a') as f:
            g = f.create_group('%s'%(name))
            g.create_dataset('pw0_C%s'%(code),data=pw_ch0)
            g.create_dataset('noise0_C%s'%(code),data=noise_a)
            g.create_dataset('pw1_C%s'%(code),data=pw_ch1)
            g.create_dataset('noise1_C%s'%(code),data=noise_b)
            g.create_dataset('t',data=tiempo)
            print("       Guardado hdf5: ",'%s.hdf5 - %s'%(folder,name))
            '''
            try:
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
                print("       Guardado hdf5: ",'%s.hdf5 - %s'%(folder,name))
            except ValueError:
                print(Fore.RED+"       Archivo *.hdf5 existente:")
                print('           %s.hdf5'%(path_o+folder))
                sys.exit(0)
            '''
          
    def ploteado(self,New_Full_Spectra_a,noise_a,New_Full_Spectra_b,noise_b,FullSpectra_a,FullSpectra_b,name,graphics_folder):
        import datetime
        tiempo = str(datetime.datetime.fromtimestamp(int(name)))
        print("Graphics_folder",graphics_folder)
        ##
        Freqs = np.linspace(-5,5,self.nc)
        Ranges = np.linspace(0,1500,self.nrange)
        ##
        min_value_a =np.min(FullSpectra_a)
        max_value_a = np.max(FullSpectra_a)
    
        min_value_b =np.min(FullSpectra_b)
        max_value_b = np.max(FullSpectra_b)
        ##
        FS_filtrado_A = New_Full_Spectra_a.toarray()
        FS_filtrado_A[FS_filtrado_A == 0] = noise_a

        FS_filtrado_B = New_Full_Spectra_b.toarray()
        FS_filtrado_B[FS_filtrado_B == 0] = noise_b

        #pw_ch0 = np.fft.ifftshift(FS_filtrado_A)
        #pw_ch1 = np.fft.ifftshift(FS_filtrado_B)

        pw_ch0 = self.spec_to_freq(FS_filtrado_A)
        #pw_ch0 = np.fft.ifftshift(FS_filtrado_A.T,axes=0)

        pw_ch1 = self.spec_to_freq(FS_filtrado_B)
        #pw_ch1 = np.fft.ifftshift(FS_filtrado_B.T,axes=0)   
        ####################
        '''     
        font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 25,
        }
        ancho = 10
        largo =16
        plt.figure(figsize=(ancho,largo))
        #plt.pcolor(Freqs, Ranges, np.log10(FullSpectra_a), cmap='jet',shading='auto')
        plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_b), cmap='jet', shading='auto',vmin=np.log10(min_value_b),vmax=np.log10(max_value_b))
        #plt.title("%s - %s"%(tiempo,"CH 1"), fontdict=font)
        plt.title("%s - %s"%(tiempo,"CH 1"), fontdict=font)
        plt.xlabel("Frequency (Hz)",fontdict=font)
        plt.ylabel("Range (km)",fontdict=font)
        plt.tick_params(axis='both', which='major', labelsize=20)
        cbar= plt.colorbar(spacing='uniform')
        cbar.ax.tick_params(labelsize=16)
        plt.ylim((0,1500))
        plt.savefig(fname=graphics_folder+name+"CH1"+".png",dpi=250)

        plt.figure(figsize=(ancho,largo ))
        plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch1,axes=0).T), cmap='jet',vmin=np.log10(min_value_b),vmax=np.log10(max_value_b))
        plt.title("%s - %s"%(tiempo,"CH 1 - Filt"), fontdict=font)
        plt.xlabel("Frequency (Hz)",fontdict=font)
        plt.ylabel("Range (km)",fontdict=font)
        cbar= plt.colorbar(spacing='uniform')
        cbar.ax.tick_params(labelsize=16)
        plt.tick_params(axis='both', which='major', labelsize=20)        
        plt.ylim((0,1500))
        plt.savefig(fname=graphics_folder+name+"CH1 - Filt"+".png",dpi=250)
        '''
        
        ###################
        font = {'family': 'serif','color':  'black','weight': 'bold','size': 18}
        font1 = {'family': 'serif','color':  'black','weight': 'bold','size': 20}

        plt.figure(figsize=(20,19))
        plt.subplot(2,2,1)
        plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_a), cmap='jet', shading='auto',vmin=np.log10(min_value_a),vmax=np.log10(max_value_a))
        cbar= plt.colorbar(spacing='uniform')
        cbar.ax.tick_params(labelsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xlabel("Frequency (Hz)",fontdict=font)
        plt.ylabel("Range (km)",fontdict=font)
        plt.ylim((0,1500))
        plt.title("%s - %s"%(tiempo,"CH0-ori"),fontdict=font1)
      
        plt.subplot(2,2,2)
        plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch0,axes=0).T), cmap='jet',vmin=np.log10(min_value_a),vmax=np.log10(max_value_a))
        cbar= plt.colorbar(spacing='uniform')
        cbar.ax.tick_params(labelsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xlabel("Frequency (Hz)",fontdict=font)
        plt.ylabel("Range (km)",fontdict=font)
        plt.ylim((0,1500))
        plt.title("%s - %s. "%(tiempo,"CH0-Filt"),fontdict=font1)
    
        plt.subplot(2,2,3)
        plt.pcolormesh(Freqs, Ranges, np.log10(FullSpectra_b), cmap='jet',vmin=np.log10(min_value_b),vmax=np.log10(max_value_b))
        cbar= plt.colorbar(spacing='uniform')
        cbar.ax.tick_params(labelsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xlabel("Frequency (Hz)",fontdict=font)
        plt.ylabel("Range (km)",fontdict=font)
        plt.ylim((0,1500))
        plt.title("%s - %s"%(tiempo,"CH1-ori"),fontdict=font1)
    
        plt.subplot(2,2,4)
        plt.pcolormesh(Freqs, Ranges, np.log10(np.fft.fftshift(pw_ch1,axes=0).T), cmap='jet',vmin=np.log10(min_value_b),vmax=np.log10(max_value_b))
        cbar= plt.colorbar(spacing='uniform') #Edicion de barra de colores
        cbar.ax.tick_params(labelsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xlabel("Frequency (Hz)",fontdict=font)
        plt.ylabel("Range (km)",fontdict=font)
        plt.ylim((0,1500))
        plt.title("%s - %s. "%(tiempo,"CH1-Filt"),fontdict=font1)

        try:
            plt.savefig(fname=graphics_folder+name+".png",dpi=200)
        except:
            os.makedirs(graphics_folder)
            print(" Directory %s created"%(graphics_folder))
            plt.savefig(fname=graphics_folder+name+".png",dpi=200)
    
        #plt.savefig('/home/soporte/Pictures/d2022241/sp01_f0/hola.png')
        plt.close()

    def read_newdata(self,path,filename,nc,value):
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
            array[array == 0] = self.read_newdata(path,filename,nc,'noise'+str(value[2])+'_C'+str(self.code))
        else:
            array = fp.get(dir_file)[()]
        fp.close()
        return array