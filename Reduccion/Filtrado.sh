#!/bin/bash

#Funcionamiento: (bash)
#./Filtrado.sh {path_in} {mode_operation} {date} {}
#./Filtrado.sh "/media/soporte/data/" 1 f0  "2021/09/27" "/media/soporte/PROCDATA/JROB/r_v/" 12
#./Filtrado.sh "/media/soporte/data/" 1 f1  "2021/09/27" "/media/soporte/PROCDATA/JROB/r_v/" 12
###########################################################
#Variable 'del' para el borrado de archivos de espectros originales. 0: No borrar *.hdf5 espectros. 1: Borrar *.hdf5 espectros
graphics_f=$7
graphics_folder="/media/igp-114/PROCDATA/Reducted/Pictures/JRO/"
graphics_folder="${graphics_f:-$graphics_folder}"
del=0
dc=0
old=1
Red=1
Plot=0
###########################################################

#./Filtrado.py -f 3.64   -path $1 -h_s 0 -code 2 -C $2 -date $3 -R 1 -del $del -path_o $4 -dc $dc -P 1 -graphics_folder="/media/soporte/RAWDATA/Pictures/Estaciones_RX/JROB/" -lo $5

# Almacenar datos de salida en diferente ruta:
#python3 r_hf.py -f 3.64 -code 0 -C 0 -date "2022/03/03" -P 0 -path "/media/soporte/Data/" -path_o "/media/soporte/PROCDATA/Reducted/JROA/" -graphics_folder "/media/soporte/PROCDATA/Reducted/Pictures/JROA/"  -lo 12 -start_h "00:00:00" -R 1

echo python3 r_hf.py -f 2.72 -path $1 -code 0 -C $2 -date $4 -path_o $5 -R $Red -del $del -dc $dc -P 1 -graphics_folder $graphics_folder -lo $6 -old $old

if [[ $3 == f0 ]]
then
python3 r_hf.py -f 2.72 -path $1 -code 0 -C $2 -date $4 -path_o $5 -R $Red -del $del -dc $dc -P $Plot -graphics_folder $graphics_folder -lo $6 -old $old
python3 r_hf.py -f 2.72 -path $1 -code 1 -C $2 -date $4 -path_o $5 -R $Red -del $del -dc $dc -P $Plot -graphics_folder $graphics_folder -lo $6 -old $old
python3 r_hf.py -f 2.72 -path $1 -code 2 -C $2 -date $4 -path_o $5 -R $Red -del $del -dc $dc -P $Plot -graphics_folder $graphics_folder -lo $6 -old $old

elif [[ $3 == f1 ]]
then
python3 r_hf.py -f 3.64 -path $1 -code 0 -C $2 -date $4 -path_o $5 -R $Red -del $del -dc $dc -P $Plot -graphics_folder $graphics_folder -lo $6 -old $old
python3 r_hf.py -f 3.64 -path $1 -code 1 -C $2 -date $4 -path_o $5 -R $Red -del $del -dc $dc -P $Plot -graphics_folder $graphics_folder -lo $6 -old $old
python3 r_hf.py -f 3.64 -path $1 -code 2 -C $2 -date $4 -path_o $5 -R $Red -del $del -dc $dc -P $Plot -graphics_folder $graphics_folder -lo $6 -old $old
fi
