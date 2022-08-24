#!/bin/bash

cd /data/
#Consultar el usuario y contraseña al area de soporte del ROJ, con acceso al radar HF
sudo mount -t cifs -o vers=1.0,username= user** ,password= password  //10.10.120.200/JRO-APP-DATA /data


echo "Montado"


#cd /home/soporte/Monitoreo/
#data_m=$(date +%m)
#En caso de analizar un mes en especifico ingresar: data_m=05
#data_m=05

#En caso de analizar un mes en especifico ingresar: data_y=2022
#data_y=$(date +%Y)
#data_y=2022

#python data_HF.py $data_m $data_y

echo "HECHO"
