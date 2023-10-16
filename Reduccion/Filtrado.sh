#!/bin/bash

#Funcionamiento: (bash)
#./Filtrado.sh {path_in} {mode_operation} {date} {}
#./Filtrado.sh "/media/soporte/data/" 1 "2021/09/27" "/media/soporte/PROCDATA/JROB/r_v/" 12

###########################################################
#Variable 'del' para el borrado de archivos de espectros originales. 0: No borrar *.hdf5 espectros. 1: Borrar *.hdf5 espectros
del=0
dc=0
old=1
Red=1
###########################################################
#Reduccion de datos: (python)
#./Filtrado.py -f {freq_exactly} -path {path} -path_o {Almacenamiento_de_salida}-code {code} -C {mode_operation} -date {date} -R {Reduced_data} -del {delete_files} -graphics_folder {finalpictures_path (ejm:'/home/soporte/Pictures/MERCED/')}
#./Filtrado.py -f 2.72216796875   -path $1 -code 0 -C $2 -date $3 -R 1 -del 1

#
#./Filtrado.py -f 2.72   -path $1 -h_s 0 -code 0 -C $2 -date $3 -R 1 -del $del -path_o $4 -dc $dc -P 1 -graphics_folder="/media/soporte/RAWDATA/Pictures/Estaciones_RX/JROB/" -lo $5 
#./Filtrado.py -f 2.72   -path $1 -h_s 0 -code 1 -C $2 -date $3 -R 1 -del $del -path_o $4 -dc $dc -P 1 -graphics_folder="/media/soporte/RAWDATA/Pictures/Estaciones_RX/JROB/" -lo $5
#./Filtrado.py -f 2.72   -path $1 -h_s 0 -code 2 -C $2 -date $3 -R 1 -del $del -path_o $4 -dc $dc -P 1 -graphics_folder="/media/soporte/RAWDATA/Pictures/Estaciones_RX/JROB/" -lo $5

#./Filtrado.py -f 3.64   -path $1 -h_s 0 -code 0 -C $2 -date $3 -R 1 -del $del -path_o $4 -dc $dc -P 1 -graphics_folder="/media/soporte/RAWDATA/Pictures/Estaciones_RX/JROB/" -lo $5
#./Filtrado.py -f 3.64   -path $1 -h_s 0 -code 1 -C $2 -date $3 -R 1 -del $del -path_o $4 -dc $dc -P 1 -graphics_folder="/media/soporte/RAWDATA/Pictures/Estaciones_RX/JROB/" -lo $5
#./Filtrado.py -f 3.64   -path $1 -h_s 0 -code 2 -C $2 -date $3 -R 1 -del $del -path_o $4 -dc $dc -P 1 -graphics_folder="/media/soporte/RAWDATA/Pictures/Estaciones_RX/JROB/" -lo $5

# Almacenar datos de salida en diferente ruta:
#python3 Filtrado.py -f 3.6498990142718 -code 2 -C $1 -date $2 -R 1 -path_o "/media/soporte/PROCDATA/Same_Struct/"

python3 r_hf.py -f 2.72 -path $1 -code 0 -C $2 -date $3 -path_o $4 -R $Red -del $del -dc $dc -P 0 -graphics_folder "/media/soporte/PROCDATA/Reducted/Pictures/JROA/" -lo $5 -old $old
python3 r_hf.py -f 2.72 -path $1 -code 1 -C $2 -date $3 -path_o $4 -R $Red -del $del -dc $dc -P 0 -graphics_folder "/media/soporte/PROCDATA/Reducted/Pictures/JROA/" -lo $5 -old $old
python3 r_hf.py -f 2.72 -path $1 -code 2 -C $2 -date $3 -path_o $4 -R $Red -del $del -dc $dc -P 0 -graphics_folder "/media/soporte/PROCDATA/Reducted/Pictures/JROA/" -lo $5 -old $old

python3 r_hf.py -f 3.64 -path $1 -code 0 -C $2 -date $3 -path_o $4 -R $Red -del $del -dc $dc -P 0 -graphics_folder "/media/soporte/PROCDATA/Reducted/Pictures/JROA/" -lo $5 -old $old
python3 r_hf.py -f 3.64 -path $1 -code 1 -C $2 -date $3 -path_o $4 -R $Red -del $del -dc $dc -P 0 -graphics_folder "/media/soporte/PROCDATA/Reducted/Pictures/JROA/" -lo $5 -old $old
python3 r_hf.py -f 3.64 -path $1 -code 2 -C $2 -date $3 -path_o $4 -R $Red -del $del -dc $dc -P 0 -graphics_folder "/media/soporte/PROCDATA/Reducted/Pictures/JROA/" -lo $5 -old $old

