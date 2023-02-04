#!/bin/bash

#Ejecucion de este programa cada fin de mes:
#./check.sh 12
path=$(pwd)
echo $path
cd /data/

sudo mount -t cifs -o vers=1.0,username=user-hf,password=5zMeOmwadpkhvfuvKijx //10.10.120.200/JRO-APP-DATA /data
echo "Montado"

path_PC="/home/soporte/"
cd $path
python3 check_files.py -path $path_PC -lo $1 -type rtdi
python3 check_files.py -path $path_PC -lo $1 -type out

sudo umount /data
