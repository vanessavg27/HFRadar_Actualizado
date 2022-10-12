#!/bin/bash

#El primer parametro de este script es la locacion
#El segundo parametro de este script es un flag de campaña
#El tercero parametro es la fecha y es opcional.

#cd $HOME/TestReduccionDatos/Plottings/
cd $HOME/Actualizado/Procesamiento/Plottings

source $HOME/TestReduccionDatos_Implementado/bin/activate
#export DISPLAY=":0.0"  #se comento por falla de generacion de plots
#El primer parametro de todos los scripts llamados es la ubicacion de la data y el segundo es la locacion de la estacion
#El tercero es un flag de campaña,1 campaña, 0 modo normal
#Se puede usar un cuarto parametro para la fecha, si se le puso fecha al script general Reduc_PLot_...
#se pasa esa fecha si no pone por defecto la fecha del dia anterior

#date_to_process=${3:-$(date -d "yesterday" +"%Y/%m/%d")} #Comentar si se desea procesar otra fecha
date_to_process="2022/07/14" # Modificar y quitar el comentario si se desea procesar otra fecha
###################Generating Moments Data#################################################3
echo "Starting Moments"
screen -S "MOMENTS_HF" -d -m ./GenerateMoments.sh "/media/soporte/PROCDATA/" $1 $2 $date_to_process
sleep 1
#Espera a que los nuevos archivos de momentos hayan terminado de generarse
COUNT=$(screen -list | grep -c "MOMENTS_HF")
while [  $COUNT != "0" ]
do
	sleep 3
	echo "Waiting for Moments to finish"
	COUNT=$(screen -list | grep -c "MOMENTS")
done
##################Plotting Parameters SNR, DOPLLER y OUT #################################################3
echo "Plotting Plottings out"
screen -S "PlottingParam_Floyd" -d -m ./Plot_RTDI_OUT.sh  "/media/soporte/PROCDATA/MomentsFloyd/" $1 $2 $date_to_process
sleep 1
COUNT=$(screen -list | grep -c "PlottingParam_Floyd")
while [  $COUNT != "0" ]
do
	sleep 3
	echo "Waiting for PlottingParam RTDI Out to finish"
	COUNT=$(screen -list | grep -c "PlottingParam")
done

###################Plotting RTDI#################################################3

echo "Plotting the new RTDI from moments"
screen -S "PLOT_RTDI" -d -m ./PLOT_RTDI.sh "/media/igp-114/PROCDATA/MomentsFloyd/" $1 $2 $date_to_process
sleep 1
COUNT=$(screen -list | grep -c "PLOT_RTDI")
while [  $COUNT != "0" ]
do
	sleep 3
	echo "Waiting for RTDI Plots to finish"
	COUNT=$(screen -list | grep -c "PLOT_RTDI")
done

cd $HOME/Actualizado/Procesamiento/SendingScripts
#cd $HOME/TestReduccionDatos_Implementado/hfschain/schainroot/source/schainpy/SendingScripts
#./sending_script.sh $1
