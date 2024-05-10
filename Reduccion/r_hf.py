#!/usr/bin/env python3.8
import datetime
import os,sys
import argparse
from multiprocessing import Process, Value, Array, Lock
from Reduc_HF import Reduccion
#from Reduc_HF_test import Reduccion

#./r_hf.py -f 2.72 -code 2 -C 1 -date "2022/08/29" -R 1 -P 1


path = os.path.split(os.getcwd())[0]
sys.path.append(path)

yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
daybefore = yesterday.strftime("%Y/%m/%d")

parser = argparse.ArgumentParser()
########################## PATH- DATA  ###################################################################################################
parser.add_argument('-path',action='store',dest='path_lectura',help='Directorio de Datos \
					.Por defecto, se esta ingresando entre comillas /media/soporte/PROCDATA/',default='/media/soporte/PROCDATA/')
########################## FRECUENCIA ####################################################################################################
parser.add_argument('-f',action='store',dest='f_freq',type=float,help='Frecuencia en Mhz 2.72 y 3.64. Por defecto, se esta ingresando 2.72216796875 ',default=2.72)
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

########################## HOUR START- SELECCION ################################################################################################
parser.add_argument('-start_h',action='store',dest='start_h',help='Seleccionar fecha si es OFFLINE se ingresa \
							la fecha con el dia deseado. Por defecto, considera el dia anterior',default='00:00:00')

########################## HOUR END- SELECCION ################################################################################################
parser.add_argument('-end_h',action='store',dest='end_h',help='Seleccionar fecha si es OFFLINE se ingresa \
							la fecha con el dia deseado. Por defecto, considera el dia anterior',default='23:59:59')

########################## LOCATION AND ORIENTATION ####################################################################################################
parser.add_argument('-lo',action='store',dest='lo_seleccionado',type=str,help='Parametro para establecer la ubicacion de la estacion de Rx y su orientacion.\
										Example: XA   ----- X: Es el primer valor determina la ubicacion de la estacion. A: Es \
										  el segundo valor determina la orientacion N45O o N45E.  \
										11: JRO-N450, 12: JRO-N45E \
												21: HYO-N45O, 22: HYO-N45E', default=11)
########################## GRAPHICS - RESULTS ###################################################################################################
parser.add_argument('-graphics_folder',action='store',dest='graphics_folder',help='Directorio de Resultados \
					.Por defecto, se esta ingresando entre comillas /home/soporte/Pictures/', default='/home/soporte/Pictures/')

parser.add_argument('-path_o',action='store',dest='path_out',help='Directorio de Datos \
					.Por defecto, se esta ingresando entre comillas /media/soporte/PROCDATA/',default='/media/soporte/PROCDATA/JROB/')

##########################  REMOVE DC ############################################################################################################
parser.add_argument('-dc',action='store',dest='remove_dc',help='Argumento para eliminar la señal DC \
		             de un espectro',default=0)
########################## DELETE SPECTRA FILES ###################################################################################################
parser.add_argument('-del',action='store',dest='delete',type=int,help='Borrado de data espectral. Data sparse. Por defecto, se esta ingresando 0',default=0)

parser.add_argument('-old',action='store',dest='old',type=int,help='Verifica si son datos antes del modo multifrecuencia',default=0)

#Parsing the options of the script
results	   = parser.parse_args()



if results.f_freq < 3:
    ngraph = 0
else:
    ngraph = 1

   
a = Reduccion(results)

#a.clean_run()