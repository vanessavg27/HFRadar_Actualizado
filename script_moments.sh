#!/bin/bash

#El primer parametro de este script es la locacion
#El segundo parametro de este script es un flag de campaña
#El tercero parametro es un flag que indica la reduccion de datos.

#./script_moments.sh 11 1 r 

#El primer parametro de todos los scripts llamados es la ubicacion de la data y el segundo es la locacion de la estacion
#El tercero es un flag de campaña,1 campaña, 0 modo normal

#se pasa esa fecha si no pone por defecto la fecha del dia anterior
#${3:-$(date -d "yesterday" +"%Y/%m/%d")}
date=$4
date_to_process=$(date -d "yesterday" +"%Y/%m/%d") #Comentar si se desea procesar otra fecha
date_to_process="${date:-$date_to_process}"
#date_to_process="2022/08/29" # Modificar y quitar el comentario si se desea procesar otra fecha
echo "date" $date_to_process

###### VARIABLES ######
ruta_procdata="/media/soporte/PROCDATA/"

#########################CONDICIONAL###############################
if [[ $3 == 'r' ]]

then
####################### REDUCE SPEC-DATA ##########################
cd $HOME/Actualizado/Reduccion/
echo "Starting Reduction!"
ruta_especreduced="/media/soporte/PROCDATA/"
#ruta_especreduced="/media/soporte/PROCDATA/MERCED/"

echo "./Reduccion_total.sh"
./Reduccion_total.sh $date_to_process

sleep 5
############ Variables ############
reduct=1
ruta_moments="/media/soporte/PROCDATA/MomentsFloyd_Filt/"
cd $HOME/Actualizado/Procesamiento/Plottings/
else
############ Variables ############
reduct=0
ruta_moments="/media/soporte/PROCDATA/MomentsFloyd/"
cd $HOME/Actualizado/Procesamiento/Plottings/

fi
########################################################### Generating Moments Data #################################################
echo "Starting Moments"
#screen -S "MOMENTS_HF" -d -m 
echo "reduct" $reduct
echo "./Moment.sh" "/media/soporte/PROCDATA/" $1 $2 $date_to_process $reduct
./Moment.sh "/media/soporte/PROCDATA/" $1 $2 $date_to_process $reduct

sleep 1
#Espera a que los nuevos archivos de momentos hayan terminado de generarse
COUNT=$(screen -list | grep -c "MOMENTS_HF")
while [  $COUNT != "0" ]
do
sleep 3
echo "Waiting for Moments to finish"
COUNT=$(screen -list | grep -c "MOMENTS")
done

######################################################### Plotting Parameters SNR, DOPLLER y OUT #################################################3
cd $HOME/Actualizado/Procesamiento/Plottings/
echo "Plotting Plottings out"
#screen -S "PlottingParam_Floyd" -d -m 
echo "./Plot_RTDI_OUT.sh" $ruta_moments $1 $2 $date_to_process
./Plot_RTDI_OUT.sh $ruta_moments $1 $2 $date_to_process
sleep 1
COUNT=$(screen -list | grep -c "PlottingParam_Floyd")
while [  $COUNT != "0" ]
do

	echo "Waiting for PlottingParam RTDI Out to finish"
	COUNT=$(screen -list | grep -c "PlottingParam")
done

###################Plotting RTDI#################################################3

echo "Plotting the new RTDI from moments"
#screen -S "PLOT_RTDI_OUT" -d -m 
./Plot_RTDI_OUT.sh $ruta_moments $1 $2 $date_to_process

sleep 1
COUNT=$(screen -list | grep -c "PLOT_RTDI")
while [  $COUNT != "0" ]
do
	sleep 3
	echo "Waiting for RTDI Plots to finish"
	COUNT=$(screen -list | grep -c "PLOT_RTDI")
done
cd $HOME/Actualizado/Procesamiento/SendingScripts
./sending_script.sh $1
