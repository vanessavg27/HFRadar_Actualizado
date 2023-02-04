#!/usr/bin/env
import os,glob,datetime
from datetime import date
import argparse
import pexpect #pip install --user  pexpect
import time
import calendar
#Ejecucion de este codigo:
#python3 check_files.py  -path /home/soporte/ -lo 12 -type out
#python3 check_files.py  -path /home/soporte/ -lo 12 -type rtdi


# En caso no se pueda enviar hacer escribir etas lineas >

#ssh -X -C wmaster@jro-app.igp.gob.pe -p 6633

#Resultado >
#The authenticity of host '[jro-app.igp.gob.pe]:6633 ([181.177.244.71]:6633)' can't be established.
#RSA key fingerprint is 24:ea:fe:d5:4e:91:8d:82:d5:7d:1f:bf:e2:0c:36:70.
#Are you sure you want to continue connecting (yes/no)? yes

debug = True
global  year
global month

def sendBySCP(Incommand, location): #"%04d/%02d/%02d 00:00:00"%(tnow.year,tnow.month,tnow.day)
	username = "wmaster"
	password = "mst2010vhf"#"123456"
	#port = "-p6633"
	host = "jro-app.igp.gob.pe"#"25.59.69.206"
	#hostdirectory = "/home/wmaster/web2/web_signalchain/data/%s/%s/%s/figures"%(place,rxname,day)
	#command = "sshfs %s %s@%s:%s %s -o nonempty "%(port,username,host,hostdirectory,mountpoint)
	command = Incommand
	if debug:
		print(command)

	console = pexpect.spawn('/bin/bash', ['-c',command])
	
	if not (location == 11 or location == 12):
		console.expect(username + "@" + host + "'s password:")			
		time.sleep(3)
		#usual response > wmaster@jro-app.igp.gob.pe's password:
		console.sendline(password)
		
	time.sleep(5)
	try:
		os.system(command) ## La magia
		#pexpect.spawn doesn't interpret shell meta characters,
		#console = pexpect.spawn(command)
		#Para reconocerlos se debe usar la siguiente linea >
		console = pexpect.spawn('/bin/bash', ['-c',command])

		if not (location == 11 or location == 12):
			console.expect(username + "@" + host + "'s password:")			
			time.sleep(3)
			#usual response > wmaster@jro-app.igp.gob.pe's password:
			console.sendline(password)
		time.sleep(5)
		#console.expect(pexpect.EOF)
		return True
	except(Exception):
		if debug:
			print("Error")
			#print(str())
		return False

def notdays(ruta_days,date_year,month):
    #print("RUTA-DAYS:",ruta_days)
    try:
        year=int(ruta_days[0][-10:-6])
        month = int(ruta_days[0][-5:-3])
    except IndexError:
        year  = int(date_year)
        month = int(month)
        
    print("Year: ",year,"Month: ", month)
    DaysInMonth = calendar.monthrange(year,month)[1]

    days_real = list(range(1,DaysInMonth+1))
    if len(ruta_days) == 0:
        days_Rx = []
        pass
    else:    
        days_Rx = [int(i[-2:]) for i in ruta_days]
        
    days_fal =[d for d in days_real if not(d in days_Rx)]
    
    daysOfYear_fal = [int(date(year,month,i).strftime('%j')) for i in days_fal]
    sorted(daysOfYear_fal)
    return daysOfYear_fal

def incompletedays(data, rxcode,out_or_figure,date_year,month):
    #data_in must be  a list like this:
    try:
        year= int(data[0][-10:-6])
        month= int(data[0][-5:-3])
    except IndexError:
        year  = date_year  
        month = month
    sorted(data)
    days_incomplete = []
    for ruta_dia in data:
        datos = glob.glob(ruta_dia+'/%s/*.%s'%(out_or_figure,out_or_figure))
        if out_or_figure == 'figures':
            #print("FIGURESS PS")
            datos = glob.glob(ruta_dia+'/%s/*.%s'%(out_or_figure,"jpeg"))
        sorted(datos)
        if rxcode in [11,12,21,22]:
            if len(datos) != 4:
                day_inc = int(ruta_dia[-2:])
                days_incomplete.append(day_inc)

        else:
            if len(datos) != 2:
                day_inc = int(ruta_dia[-2:])
                days_incomplete.append(day_inc)
    sorted(days_incomplete)

    daysOfYear_inc = [int(date(year,month,i).strftime('%j')) for i in days_incomplete]
    return daysOfYear_inc

def File_writer(dias_faltantes,station_type,datatype,year,rxcode):
    print("YEAR Fx: ",year)
    Tx_s = list(dias_faltantes.keys())
    print("Lista de entrada: ",Tx_s)
    cont=0
    #Comparacion de elementos faltantes en las listas que si lo contienen
    for t in Tx_s:
        print("")
        print("Para ",t)
        Tx_v = Tx_s.copy()
        Tx_v.remove(t)
        for t_v in Tx_v:
            print("revisar,",t_v)
            if cont == 0:
                days_f =  [d for d in dias_faltantes[t] if not (d in dias_faltantes[t_v])]
                cont += 1
            else:
                days_f +=  [d for d in dias_faltantes[t] if not (d in dias_faltantes[t_v])]
                cont += 1
            
    days_f= sorted(list(set(days_f)))
    print("Days_f: ",days_f)
    #Elegir el maximo
    for d in days_f:
        print("d general: ",d)
        for i in range(len(Tx_s)):
            print("")
            print("d:",d, not (d in dias_faltantes[Tx_s[i]]))
            print(Tx_s[i],"BEFORE",dias_faltantes[Tx_s[i]],sep=" ")
            if not (d in dias_faltantes[Tx_s[i]]):
                #print("Lista de dia faltante: ",dias_faltantes[Tx_s[i]])
                cont_n=-1
                for j in dias_faltantes[Tx_s[i]]:
                    
                    indice = dias_faltantes[Tx_s[i]].index(j)
                    a_None = [indice for indice, dato in enumerate(dias_faltantes[Tx_s[i]]) if dato == None]
                    
                    try:
                        if j > d and dias_faltantes[Tx_s[i]][indice-1] < d:
                        
                            #indice = dias_faltantes[Tx_s[i]].index(j)
                            #print("\n")
                            #print("Dia faltante: ",d)
                            dias_faltantes[Tx_s[i]].insert(indice,None)
                            #print(Tx_s[i]," AFTER: ",dias_faltantes[Tx_s[i]],sep=" ")
                            break
                    except:
                        if j == None:
                            cont_n += 1

                            if dias_faltantes[Tx_s[i]][a_None[cont_n]+1] == None:
                                continue
                            else:
                                if dias_faltantes[Tx_s[i]][a_None[cont_n]+1] > d:
                                    dias_faltantes[Tx_s[i]].insert(a_None[cont_n]+1,None)
                                    #print(Tx_s[i]," AFTER: ",dias_faltantes[Tx_s[i]],sep="\n")
                                    break
                                else:
                                    continue
    #print(rxcode)
    print(dias_faltantes)

    for i in range(len(Tx_s)):
        print(Tx_s[i],":",len(dias_faltantes[Tx_s[i]]))
        for j in range(len(dias_faltantes[Tx_s[i]])):
            if dias_faltantes[Tx_s[i]][j] == None:
                continue
            dias_faltantes[Tx_s[i]][j] = year*1000+dias_faltantes[Tx_s[i]][j]
    try:
        os.remove("Doys_not_sent_%s_%s"%(datatype,station_type))
    except:
        os.system("touch Doys_not_sent_%s_%s"%(datatype,station_type))
    #os.remove("Doys_not_sent_%s_%s"%(datatype,station_type))
    lon = len(dias_faltantes[Tx_s[0]])
    
    with open("Doys_not_sent_%s_%s"%(datatype,station_type), "w") as f:
        print(list(dias_faltantes.keys()))
        codigos = [0,1,2]
        for i in range(lon):
            lista = [ dias_faltantes[t][i] for t in list(dias_faltantes.keys()) ]
            #print(lista)
            if rxcode in [11,12,21,22]: #Estacion doble
                f.write(str(lista[0])+','+str(lista[1])+','+str(lista[2])+"\n") #File contains
            else:  #Estacion simple
                lista2 = []
                for i in codigos:
                    if lista[2*i+0] ==  lista[2*i+1] == None:
                        lista2.append(lista[2*i+0])
                    elif lista[2*i+0] != None:
                        lista2.append(lista[2*i+0])
                    elif lista[2*i+1] != None:
                        lista2.append(lista[2*i+1])
                
                f.write(str(lista2[0])+','+str(lista2[1])+','+str(lista2[2])+"\n") #File contains

def dias_faltantes(ruta_base, rxcode,rxname,datatype,year=2022):
    #station_rx = rxname
    station_tx =  ['HFTXANCON','HFBTXANCON','HFTXSICAYA','HFBTXSICAYA','HFTXICA','HFBTXICA']
    station_orientation = "A" if rxcode % 10 == 1 else "B"
    out_or_figure = "out" if datatype == "out" else "figures"
    AorB = rxcode%2

    Days_off_tx = {}
    Days_inc_tx = {}
    
    if year !=  2022:
        date_year  = date.today().year
        date_month = date.today().month
        date_day   = date.today().day
    else:
        date_year  = year
        date_month = 12
        #date_day   = 
    #date_month = date.today().month
    date_day   = date.today().day
    if rxcode in [11,12,21,22]:
        if not(AorB):
            tx_s = [ i for i in station_tx if i[:3]== 'HFB']
        else:
            tx_s = [ i for i in station_tx if i[:3]== 'HFT']
    else:
        tx_s = station_tx
        station_orientation = "A"
    
    dias_faltantes = {}

    for tx in tx_s:
        print("*****",tx,"*****")
        ruta_tx = ruta_base+rxname+"/"+tx+'/'+str(date_year)+"/"
        days_off=[]
        days_incomplete = []
        print("ruta_tx",ruta_tx)
        for month in range(1,date_month+1):
            if len(str(month)) == 1:
                month = '0'+str(month)
            else:
                month =str(month)
            print("")
            print("MES: ",month)
            ruta_mes  = ruta_tx+month+'/'
            datos_dias=glob.glob(ruta_mes+'/*')
            sorted(datos_dias)
            #print("datos_dias",datos_dias)
            days_off += notdays(datos_dias,date_year,month)
            Days_off_tx[tx]=days_off

            days_incomplete += incompletedays(datos_dias, rxcode,out_or_figure,date_year,month)
            Days_inc_tx[tx] = days_incomplete
            Days_off_tx[tx] = sorted(days_off + days_incomplete)

    print("")
    print("Dias de datos incompletos:",Days_off_tx)
    File_writer(Days_off_tx,station_orientation,datatype,date_year,rxcode)



################################### INGRESANDO CODIGO 0 o 2 ##############################################################
parser = argparse.ArgumentParser()
parser.add_argument('-code',action='store',dest='code_seleccionado',help='Code para generar Spectro off-line 0,1,2',default=0)
parser.add_argument('-lo',action='store',dest='localstation', default =11,help='Codigo de estacion 11 JRO A, 12 JRO B, 21 HYO A, 22 HYO B, etc..')
parser.add_argument('-path',action='store',dest='path',default="/home/soporte/Pictures/", help='Directorio principal donde estan los resultados a enviar')
parser.add_argument('-type',action='store',dest='type',default="params" ,help='Determines the type of data to be sent. Could be "out"-"params"-"rtdi".')
parser.add_argument('-year',action='store',dest='year',default=2022,help='YEAR to analyze')

results    = parser.parse_args()
code       = int(results.code_seleccionado)
rxcode     = int(results.localstation)
PATH       = str(results.path)
datatype   = str(results.type)
year       = int(results.year)


###########################################################################################################################
tdstr = datetime.date.today()
str1 = tdstr + datetime.timedelta(days=-1)
yesterday = str1.strftime("%Y%j")

ruta_base = "/data/hfdatabase/webhf/web_rtdi/data/"

dlist = []
file_lines = []
station_type = "A" if rxcode % 10 == 1 else "B"
print("***",datatype,station_type)

location_dict = {11:"JRO_A", 12: "JRO_B", 21:"HYO_A", 22:"HYO_B", 31:"MALA",
	41:"MERCED", 51:"BARRANCA", 61:"OROYA"}

if rxcode == 11 or rxcode == 12 :
	rxname = 'JRO'
if rxcode == 21 or rxcode == 22 :
	rxname = 'HUANCAYO'
if rxcode == 31 or rxcode == 32 :
	rxname = 'MALA'
if rxcode == 41 or rxcode == 42 :
	rxname = 'MERCED'
if rxcode == 51 or rxcode == 52 :
	rxname = 'BARRANCA'
if rxcode == 61 or rxcode == 62 :
	rxname = 'OROYA'


###Funcion para crear los archivos de dias faltantes
dias_faltantes(ruta_base, rxcode,rxname,datatype,year)

'''
with open("Doys_not_sent_%s_%s"%(datatype,station_type),"r") as f:
    # The Doys_not_sent file have data about the doys that hasn't been sent, each line represents a doys
    # And each line has the same doy or None for each code, None represents that the data hasn't
    # been sent yet for that code.
    for line in f.readlines():
        file_lines.append(line[:-1].split(","))
        doy_or_none = file_lines[-1][code]
        if doy_or_none != "None":
            dlist.append(doy_or_none)

if yesterday not in [doy for line in file_lines for doy in line]:
    dlist.append(yesterday)
    file_lines.append([yesterday]*3)

if rxcode in [11, 12, 21, 22]:
    station_orientation = "A" if rxcode % 10 == 1 else "B"
else:
    station_orientation = "A"

if rxcode in [11, 12, 21, 22]:
    lo = rxcode
else:
    lo = int(rxcode/10)*10+1

print("PATH: ",PATH)

if datatype == "params":
    graph_freq0=PATH+"GRAPHICS_SCHAIN_%s/"%location_dict[lo]+'sp'+str(code)+'1_f0'
    graph_freq1=PATH+"GRAPHICS_SCHAIN_%s/"%location_dict[lo]+'sp'+str(code)+'1_f1'
else:
    if PATH == "/home/soporte/":
        graph_freq0 = PATH+"Pictures/RTDI_%s/graphics_schain/"%station_orientation + 'sp'+str(code)+'1_f0'
        graph_freq1 = PATH+"Pictures/RTDI_%s/graphics_schain/"%station_orientation + 'sp'+str(code)+'1_f1'
    else:
        graph_freq0 = PATH+"RTDI_%s/graphics_schain/"%station_orientation + 'sp'+str(code)+'1_f0'
        graph_freq1 = PATH+"RTDI_%s/graphics_schain/"%station_orientation + 'sp'+str(code)+'1_f1'

print("")
print("GRAPH_FREQ0: ",graph_freq0)
print("GRAPH_FREQ1: ",graph_freq1)
print("")
str_datatype = "*/" if datatype == "params" else ""

web_type = "web_signalchain" if datatype == "params" else "web_rtdi"
letter = "H" if datatype == "out" else "" # This adds the H.
extension = "out" if datatype == "out" else "jpeg"
out_or_figures = "out" if datatype == "out" else "figures"

for file in dlist[:]: # Se usa una copia porque dlist puede ser modificado dentro del bucle
    data_written_0 = True #Flag that represents data_written
    data_written_1 = True
    doy = 'd'+file

    #jpg_files = glob.glob("%s/%s/%s*.jpeg"%(graph_freq1, doy,str_datatype))
    arch_files_0= glob.glob("%s/%s/%s*.%s"%(graph_freq0, doy,str_datatype,extension))
    arch_files_str= "%s/%s/%s*.%s"%(graph_freq0, doy,str_datatype,extension)
    arch_files_0.sort()
    arch_files_1= glob.glob("%s/%s/%s*.%s"%(graph_freq1, doy,str_datatype,extension))
    arch_files_1.sort()

    print("DATA FREQ0:",arch_files_0)
    print("DATA FREQ1:",arch_files_1)

    #jpg_files.sort()
    #arch_files.sort()
    #print("LISTA",arch_files)
    #print("%s/%s/%s*.jpeg"%(graph_freq1, doy,str_datatype))
    #if len(jpg_files) is 0:
    if len(arch_files_0) is 0 and len(arch_files_1) is 0:
        print('No hay RESULTADOS en la carpeta %s !!'%(arch_files_str))
        print("")
        continue
        #file_1=os.path.basename(jpg_files[0])[offset:]
    if datatype == "params":
        #file_1 = jpg_files[0][:-1].split("/")[-3]
        file_1 = arch_files_1[0][:-1].split("/")[-3]
    else:
        #file_1 = jpg_files[0][:-1].split("/")[-2]
        if arch_files_0 == []:
            file_1 = arch_files_1[0][:-1].split("/")[-2]
        elif arch_files_1 == []:
            file_1 = arch_files_0[0][:-1].split("/")[-2]
        else:
            file_1 = arch_files_0[0][:-1].split("/")[-2]

    file_1 = file_1[1:]
    print("FILE: ",file_1)

    YEAR=int (file_1[0:4])
    DAYOFYEAR= int(file_1[4:7])
    DAYOFYEAR_str = file_1[4:7]
    d = datetime.date(YEAR, 1, 1) + datetime.timedelta(DAYOFYEAR - 1)
    print("(1) Determinando dia a enviar.")
    #print('d.strftime("%Y/%m/%d")>', d.strftime("%Y/%m/%d"))
    print("Fecha: ",d.strftime("%Y/%m/%d"))
    MONTH = d.strftime("%m")
    DAY = d.strftime("%d")
    AorB = rxcode%2 # Si es par(B) AorB sera 0, si es impar(A) AorB sera 1.

    if rxcode == 11 or rxcode == 12 :
        rxname = 'JRO'
    if rxcode == 21 or rxcode == 22 :
        rxname = 'HUANCAYO'
    if rxcode == 31 or rxcode == 32 :
        rxname = 'MALA'
    if rxcode == 41 or rxcode == 42 :
        rxname = 'MERCED'
    if rxcode == 51 or rxcode == 52 :
        rxname = 'BARRANCA'
    if rxcode == 61 or rxcode == 62 :
        rxname = 'OROYA'


    if code ==0 and AorB == 1:
        station_name="HFTXANCON"
    if code ==0 and AorB == 0:
        station_name="HFBTXANCON"
    if code ==1 and AorB == 1:
        station_name="HFTXSICAYA"
    if code ==1 and AorB == 0:
        station_name="HFBTXSICAYA"
    if code ==2 and AorB == 1:
        station_name="HFTXICA"
    if code ==2 and AorB == 0:
        station_name="HFBTXICA"

    #remote_folder="/home/wmaster/web2/web_rtdi/data/JRO/%s/%s/%s/%s/figures/"%(station_name, YEAR, MONTH,DAY)
    remote_folder="/home/wmaster/web2/%s/data/%s/%s/%s/%s/%s/%s/"%(web_type,rxname,station_name, YEAR, MONTH,DAY, out_or_figures)
    print("")
    print("(2) Direccion de llegada de datos.")
    print("Remote_folder: %s"%(remote_folder))
    print("")
    #Primer Comando, generar carpeta de destino.

    command_1="ssh wmaster@jro-app.igp.gob.pe -p 6633 mkdir -p %s"%(remote_folder)
    print("HOLITAS",sendBySCP(command_1, rxcode))
    if sendBySCP(command_1, rxcode):
        print('Carpeta Creada')


    #os.system("ssh wmaster@jro-app.igp.gob.pe -p 6633 %s "%(command_1))

    #Segundo comando, pasar las imagenes necesarias para la freq 0
    if arch_files_0 != []:
        print(" (3) Enviando resultados frecuencia 0")
        temp_command0 = "scp -r -P 6633 %s/%s/%s%s%s%s%s*.%s wmaster@jro-app.igp.gob.pe:%s"%(graph_freq0,doy,str_datatype,letter,YEAR, DAYOFYEAR_str, rxcode, extension, remote_folder)

        if sendBySCP(temp_command0, rxcode):
            print(' -- Datos Enviados F0')
        else:
            data_written_0=False
            print("Datos no se enviaron.")
    else:
        data_written_0 = False

	#Segundo comando, pasar las imagenes necesarias para la freq 1					AQUI
    if arch_files_1 != []:
        #temp_command = "scp -r -P 6633 %s/%s/*.jpeg wmaster@jro-app.igp.gob.pe:%s"%(graph_freq1,doy,remote_folder)
        print("(4) Enviando resultados frencuencia 1") #DAYOFYEAR_str
        temp_command1 = "scp -r -P 6633 %s/%s/%s%s%s%s%s*.%s wmaster@jro-app.igp.gob.pe:%s"%(graph_freq1,doy,str_datatype,letter,YEAR, DAYOFYEAR_str, rxcode, extension, remote_folder)
        if sendBySCP(temp_command1, rxcode):
	        print(' -- Datos enviados F1 ')
        else:
            data_written = False
            print("Datos no se enviaron.")
    else:
        data_written_1 = False
    ##Checks if all data where sent, if not save the doy in the file what contains all doy not sent yet.
    if data_written_0 and data_written_1:
        #dlist.remove(file)
        index = [file_lines[i][code] for i in range(len(file_lines))].index(file)
        file_lines[index][code] = "None"
        if file_lines[index].count("None") == 3:
            file_lines.pop(index)

print("\n\n\n")
os.remove("Doys_not_sent_%s_%s"%(datatype,station_type))
with open("Doys_not_sent_%s_%s"%(datatype,station_type), "w") as f:
    for line in file_lines:
        f.write(line[0]+','+line[1]+','+line[2]+"\n") #File contains date in 2019200 format yeardoy
'''