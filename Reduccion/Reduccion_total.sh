#!/bin/bash

echo "Iniciando Filtrado y Reduccion de datos HF ..."

echo "Iniciando F0"
screen -S "Reduccion_F0" -d -m ./Filtrado.sh "/media/soporte/PROCDATA/" 0 f0 "2023/02/01" "/media/soporte/PROCDATA/Reducted/JROB/" 12

for i in {1..4}
do
   echo -n "."
   sleep 1s
done

echo "Iniciando F1"
screen -S "Reduccion_F1" -d -m ./Filtrado.sh "/media/soporte/PROCDATA/" 0 f1 "2023/02/01" "/media/soporte/PROCDATA/Reducted/JROB/" 12

