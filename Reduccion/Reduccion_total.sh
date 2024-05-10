#!/bin/bash

#############################################################################################
#MODE_OPERATION: 0 -> Normal Mode. 1 -> Campaign Mode.
#lo={JROA:11, JROB:12, HYOA:21, HYOB:22, MALA:31, MERCED:41, BARRANCA:51,LA OROYA:61}
#path_in="/media/igp-114/PROCDATA/"


#./Reduccion_total.sh [Date:"YYYY/MM/DD"] [MODE_OPERATION] [lo] [path_in] [path_out_reducted] [optional:graphics_folder]
#./Reduccion_total.sh "2023/01/01" 0 11 path_in path_out_reducted
#./Reduccion_total.sh 2021/10/8 0 12 /media/soporte/data/ /home/soporte/Desktop/Reducted/BARRANCA/ /home/soporte/Desktop/Reducted/Pictures/BARRANCA/

#./Filtrado.sh [path_in] [MODE_OPERATION] f0 [Date:"YYYY/MM/DD"] [path_out_reducted] [lo] [optional:graphics_folder]
#./Filtrado.sh [path_in] [MODE_OPERATION] f1 [Date:"YYYY/MM/DD"] [path_out_reducted] [lo] [optional:graphics_folder]
#############################################################################################

echo "Iniciando Filtrado y Reduccion de datos HF ..."

echo "Iniciando F0"

echo "./Filtrado.sh" $4 $2 f0 $1 $5 $3 $6
screen -S "Reduccion_F0" -d -m ./Filtrado.sh $4 $2 f0 $1 $5 $3 $6

for i in {1..4}
do
   echo -n "."
   sleep 1s
done

echo "Iniciando F1"
echo "./Filtrado.sh" $4 $2 f1 $1 $5 $3 $6
screen -S "Reduccion_F1" -d -m ./Filtrado.sh $4 $2 f1 $1 $5 $3 $6
COUNT=$(screen -list | grep -c "Reduccion_F1")
while [  $COUNT != "0" ]
do
sleep 3
echo "Waiting for Reducting"
COUNT=$(screen -list | grep -c "Reduccion_F1")
done
