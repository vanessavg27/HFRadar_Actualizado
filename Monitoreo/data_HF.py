#!/usr/bin/python
import os, glob
import argparse
from datetime import datetime
import datetime
import matplotlib
import numpy as np
from calendar import monthrange
import matplotlib.pyplot as plt
import numpy as np
import datetime,  time
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import smtplib, sys
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

parser = argparse.ArgumentParser(description="Process dates")
parser.add_argument('month',type=str,action="store",help="Month to evaluate")
parser.add_argument('year',type=str,action="store",help="Year to evaluate")
#fecha.add_argument('-date',dest='date',type=str)
fecha =parser.parse_args()

date_meses = datetime.date.today()
#date_year="2022"

date_year = fecha.year

date_mes = date_meses.strftime("%m")
#date_mes = '05'
date_mes = fecha.month

num_dia = monthrange(2022,int(date_mes))[1]
if date_mes == date_meses.strftime("%m"):
        num_dia = int(date_meses.strftime("%d"))
print("date_mes",date_mes)
#ruta_base = "/home/wmaster/web2/web_rtdi/data/"
ruta_base = "/data/hfdatabase/webhf/web_rtdi/data/"
station_rx = ['JRO', 'HUANCAYO','MALA','MERCED','BARRANCA','OROYA']
station_tx = station_tx = ['HFTXANCON','HFBTXANCON','HFTXSICAYA','HFBTXSICAYA','HFTXICA','HFBTXICA']
dicta = {}
dicta['JROA']={}
dicta['JROB']={}
dicta['HYOA']={}
dicta['HYOB']={}
dicta['MALA']={}
dicta['MERCED']={}
dicta['BARRANCA']={}
dicta['OROYA']={}


cantidad1 = []
for rx in station_rx:
        print("")
        print("Rx",rx)
        dicta[rx] = {}
        lista_date = []
        list_doubleA_station=[]
        list_doubleB_station = []
        for tx in station_tx:
                meses = ruta_base+rx+'/'+tx+'/'+date_year+'/'+date_mes+'/'
                for dia in sorted(glob.glob(meses+'/*')):
                        #num_dia = len(glob.glob(meses+'/*'))
                        data_out = 0
                        fecha = os.path.basename(dia)+'/'+os.path.basename(date_mes)+'/'+date_year

                        for out in sorted(glob.glob(dia+'/out/*')):
                                data_o = len(open(out).readlines())
                                data_out += data_o
                        #print("")
                        #print(tx,"data_out:",data_out)
                        if rx == 'JRO' or rx == 'HUANCAYO':
                                if tx in ['HFTXANCON','HFTXSICAYA','HFTXICA']:
                                        data_out_double=data_out
                                        list_doubleA_station.append([fecha,tx,data_out_double])
                                if tx in ['HFBTXANCON','HFBTXSICAYA','HFBTXICA']:
                                        data_out_double=data_out
                                        list_doubleB_station.append([fecha,tx,data_out_double])
                        else:

                             	lista_date.append([fecha,tx,data_out])
        #dicta[rx] = lista_date
        #print dicta
        print("")
        print("LISTA_DATE: ",lista_date)
        print("LIST-DOBLE-A: ",list_doubleA_station)
        print("LIST-DOBLE-B: ",list_doubleB_station)
        print("")
        dias_mes = []
        cantidad = []
        cantidad_a = []
        cantidad_b = []
        totales = []
        print("type_numdia: ",type(num_dia),num_dia)

        for i in range(1,num_dia):
                td = datetime.timedelta(days=int("%s"%(i)))
                if date_mes == date_meses.strftime("%m"):
                        j =  datetime.date.today()-td
                        l = j.strftime("%d/%m/%Y")
                else:
                     	#from datetime
                        time_data= str(num_dia)+"/"+date_mes+"/"+date_year[2:]
                        j= datetime.datetime.strptime(time_data,"%d/%m/%y").date() - datetime.timedelta(days=int("%s"%(i)))
                        l=j.strftime("%d/%m/%Y")

                print("L Dia: ",l)
                n = j.strftime("%d")
                d = 0
                d_A = 0
                d_B = 0
                #print "lista_datex2: ",lista_date

                if rx == 'JRO' or rx == 'HUANCAYO':

                        for data_A in list_doubleA_station:
                                if l == data_A[0]:
                                        #print("Programa entro, rx :",rx,"-A")
                                        print("Analizando estacion ",rx,"-A")
                                        #print "DATA_A: ",data_A
                                        d_A += int(data_A[2])

                        porcentaje_a = int(100 * int(d_A)/17220)
                        cantidad_a.append(int(porcentaje_a))
                        print("Porcentaje 6 folderes",rx,"-A", porcentaje_a,"%")

                        for data_B in list_doubleB_station:
                                if l == data_B[0]:
                                        #print("Programa entro, rx :",rx,"-B")
                                        print("Analizando estacion ",rx,"-B")
                                        #print "DATA_B: ",data_B
                                        d_B += int(data_B[2])
                        porcentaje_b = int(100*int(d_B)/17220)
                        cantidad_b.append(int(porcentaje_b))
                        print("List-porcentaje_b: ",cantidad_b,"(cantidad_b)")
                        print("Porcentaje 6 folderes",rx,"-B", porcentaje_b,"%")
                else:
                    for data in lista_date:
                        if l == data[0]:
                            print("Programa entro, rx :",rx)
                            print("Analizando estacion simple", data)
                            d += int(data[2])
                    porcentaje = int(100 * int(d)/17220)
                    print("D :",int(d))
                    cantidad.append(int(porcentaje))
                    print("Porcentaje 6 folderes", rx,  porcentaje,"%")
                    print("List-porcentaje_simple: ",cantidad,"(cantidad)")


        print("c_a*",cantidad_a)
        print("c_b*",cantidad_b)
        print("c_*",cantidad)


        #print "listando-Removiendo", lista_date

        #print list_doubleA_station

        #print list_doubleA_station

        #print "Estacion", rx, "Sin lista_date", lista_date
        #dias_mes.reverse()
        print("Iniciando guardado")
        cantidad.reverse()
        cantidad_a.reverse()
        print(cantidad_a)
        cantidad_b.reverse()
        print(cantidad_b)
        print("*",map(int,cantidad))
        today = datetime.date.today()

        if rx == 'JRO':
                
                dicta['JROA']["%s/%s"%(date_mes,date_year)]= cantidad_a

                dicta['JROB']["%s/%s"%(date_mes,date_year)]= cantidad_b

        elif rx ==  'HUANCAYO':
               
                dicta['HYOA']["%s/%s"%(date_mes,date_year)]= cantidad_a
              
                dicta['HYOB']["%s/%s"%(date_mes,date_year)]= cantidad_b
               
        else:
            
            dicta[rx]["%s/%s"%(date_mes,date_year)]= cantidad

print("DICCIONARIO: ",dicta)


#
#
#
month_name = datetime.datetime.strptime(date_mes,"%m").strftime("%B")
today = datetime.datetime.today()
today_1 = today.strftime("%B")
dates = today.strftime("%Y%j")
revision = today.strftime("%c")
#print(today_1)
print(month_name)
date = date_mes+"/"+date_year
#date = "06/2022"

ruta_local = "/home/wmaster/Downloads/%s"%('JROa')


cantidad = [dicta['JROA'][date],dicta['JROB'][date],dicta['HYOA'][date],dicta['HYOB'][date],dicta['MALA'][date],dicta['MERCED'][date],dicta['BARRANCA'][date],dicta['OROYA'][date]]
print("BEFORE: ",cantidad)
#cantidad = [dicta['OROYA'][date],dicta['BARRANCA'][date],dicta['MERCED'][date],dicta['MALA'][date],  dicta['JROA'][date],dicta['JROB'][date],dicta['HYOA'][date],dicta['HYOB'][date],]
cantidad.reverse()
print("AFTER: ",cantidad)


dias= list(np.arange(0,num_dia))
station = list(np.arange(0.5,9))

#plt.figure(figsize=(10,4))
fig, ax = plt.subplots(figsize = (15,6))
#cmap = ListedColormap(['red', 'yellow', 'dodgerblue'])
#cmap = ListedColormap(['mediumblue','yellow','deepskyblue'])
cmap = ListedColormap(['mediumblue','mediumblue','deepskyblue'])
#cmap = ListedColormap(['mediumblue','deepskyblue'])
plt.pcolormesh(dias,station,cantidad, cmap= cmap,  edgecolors = 'k', linewidths=0.2)

#red_patch = mpatches.Patch(color="dodgerblue", label='Normal')
#red_patch1 = mpatches.Patch(color='red', label='Sin acceso a red ')
#red_patch2 = mpatches.Patch(color='yellow', label='Falla de  sistema (ADQ o PROC)')

red_patch = mpatches.Patch(color="deepskyblue", label='Normal')
#red_patch1 = mpatches.Patch(color='yellow', label='Sin acceso a red ')
red_patch2 = mpatches.Patch(color='mediumblue', label='Falla de  sistema Rx-HF')
#fig.legend(handles=[red_patch,red_patch1,red_patch2],loc = 1, borderaxespad = 0.000)
fig.legend(handles=[red_patch,red_patch2],loc = 1, borderaxespad = 0.000)

#cbar = plt.colorbar()
#cbar.set_label("Horas de Operacion", rotation = -270)
plt.title("Proyecto HF, %s-%s"%(month_name,date_year))
#ax.set_yticklabels([".","JROA","JROB","HYOA","HYOB","MALA","MERCED","BARRANCA","OROYA"])
ax.set_yticklabels([".","OROYA","BARRANCA","MERCED","MALA","HYOB","HYOA","JROB","JROA"])
#plt.xlabel("Days - %s"%(today_1))
plt.xlabel("Days - %s %s"%(month_name,date_year))
plt.ylabel("Station")
#plt.savefig("Proyecto-HF-%s"%(today_1))
plt.savefig("Proyecto-HF-%s_%s"%(month_name,date_year))

plt.show()

#filename = "Hola Karim\n" "Se adjunta la operatividad del sistema Rx del mes de %s\n"%(today_1)
filename = "Hola Karim\n" "Se adjunta la operatividad del sistema Rx del mes de %s\n"%(month_name)

if True:
    sender = 'hfradar463@gmail.com'
    proyect = 'Proyecto Radar HF'
    password ='RadarHF-ROJ'
    receivers = ['lvilcatoma@igp.gob.pe' ,'kkuyeng@igp.gob.pe','roj-op02@igp.gob.pe']

    subject     = 'Operatividad - Sistema Radar HF'
    SMTP_SERVER = 'smtp.gmail.com'
    SMTP_PORT   = 465
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['To']      = 'lvilcatoma@igp.gob.pe'
    msg['To']      = 'kkuyeng@igp.gob.pe'
    msg['To']      = 'roj-op02@igp.gob.pe'
    msg['From']    = proyect
    msg['Message'] = filename
    part = MIMEText('text', "plain")
    part.set_payload(filename)
    msg.attach(part)
    file = open("Proyecto-HF-%s_%s.png"%(month_name,date_year), "rb")
    attach_image = MIMEImage(file.read())
    attach_image.add_header('Content-Disposition', 'attachment; filename = "Proyecto-HF.png"')
    msg.attach(attach_image)
    try:
        session= smtplib.SMTP_SSL(SMTP_SERVER,SMTP_PORT)
        session.login(sender,password)
        session.sendmail(sender,receivers,msg.as_string())
        session.quit()
        print ("Mensaje Enviado Luchito")
    except smtplib.SMTPException:
        print ("Error: No se envio")
########################################################
print("* * * Fin de verificacion * * * ")


#print "lista2", cantidad1
#print type(cantidad1)
