#!/bin/bash

cd /data/

sudo mount -t cifs -o vers=1.0,username=user-hf,password=5zMeOmwadpkhvfuvKijx //10.10.120.200/JRO-APP-DATA /data
echo "Montado"
cd /home/soporte/Actualizado/Monitoreo/
data_m=$(date +%m)
#En caso de analizar un mes en especifico ingresar: data_m=05
#data_m=05

#En caso de analizar un mes en especifico ingresar: data_y=2022
data_y=$(date +%Y)
#data_y=2022

python3 data_HF.py $data_m $data_y
#python3 data_HF.py 08 2022
echo "HECHO"

sudo umount /data
