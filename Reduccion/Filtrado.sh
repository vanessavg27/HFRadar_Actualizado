#!/bin/bash

#Modo Campaña
python3 Filtrado.py -f 2.72 -code 0 -C $1 -date $2 -path_o "/media/soporte/PROCDATA/Same_Struct/" -clean 1 -P 1
python3 Filtrado.py -f 2.72 -code 1 -C $1 -date $2 -path_o "/media/soporte/PROCDATA/Same_Struct/" -clean 1 -P 1
python3 Filtrado.py -f 2.72 -code 2 -C $1 -date $2 -path_o "/media/soporte/PROCDATA/Same_Struct/" -clean 1 -P 1

python3 Filtrado.py -f 3.64 -code 0 -C $1 -date $2 -path_o "/media/soporte/PROCDATA/Same_Struct/" -clean 1 -P 1
python3 Filtrado.py -f 3.64 -code 1 -C $1 -date $2 -path_o "/media/soporte/PROCDATA/Same_Struct/" -clean 1 -P 1
python3 Filtrado.py -f 3.64 -code 2 -C $1 -date $2 -path_o "/media/soporte/PROCDATA/Same_Struct/" -clean 1 -P 1

#-path_o "/media/soporte/PROCDATA/Same_Struct/"
