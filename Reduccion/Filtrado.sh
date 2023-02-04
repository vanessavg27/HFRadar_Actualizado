#!/bin/bash

#Funcionamiento: (bash)
#./Filtrado.sh {path} {mode_operation} {date}
#./Filtrado.sh "/media/soporte/DATA1/" 1 "2022/08/29" "/media/soporte/PROCDATA/MERCED/"

###########################################################
#Variable del para el borrado de archivos de espectros originales
del=0
###########################################################
#Reduccion de datos: (python)
#./Filtrado.py -f {freq_exactly} -path {path} -path_o {Almacenamiento_de_salida}-code {code} -C {mode_operation} -date {date} -R {Reduced_data} -del {delete_files}
#./Filtrado.py -f 2.72216796875   -path $1 -code 0 -C $2 -date $3 -R 1 -del 1

#
./Filtrado.py -f 2.72216796875   -path $1 -h_s 0 -code 0 -C $2 -date $3 -R 1 -del $del -path_o $4 
./Filtrado.py -f 2.7221646392718 -path $1 -h_s 0 -code 1 -C $2 -date $3 -R 1 -del $del -path_o $4 
./Filtrado.py -f 2.7221712982282 -path $1 -h_s 0 -code 2 -C $2 -date $3 -R 1 -del $del -path_o $4 

./Filtrado.py -f 3.64990234375   -path $1 -h_s 0 -code 0 -C $2 -date $3 -R 1 -del $del -path_o $4 
./Filtrado.py -f 3.6499056732282 -path $1 -h_s 0 -code 1 -C $2 -date $3 -R 1 -del $del -path_o $4 
./Filtrado.py -f 3.6498990142718 -path $1 -h_s 0 -code 2 -C $2 -date $3 -R 1 -del $del -path_o $4 

# Almacenar datos de salida en diferente ruta:
#python3 Filtrado.py -f 3.6498990142718 -code 2 -C $1 -date $2 -R 1 -path_o "/media/soporte/PROCDATA/Same_Struct/"
