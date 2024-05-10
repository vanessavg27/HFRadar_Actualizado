#!/bin/bash

#################################################################
#El primer parametro de este script es la locacion
#El segundo parametro de este script es un flag de campaña
#El tercero parametro es un flag que indica la reduccion de datos.
#lo={JROA:11, JROB:12, HYOA:21, HYOB:22, MALA:31, MERCED:41, BARRANCA:51,LA OROYA:61}

#Automatico
#./script_moments.sh 11 1 r  
#Manual
#./script_moments.sh 12 0 r "2021/10/8"
#################################################################

date=$4
date_to_process=$(date -d "yesterday" +"%Y/%m/%d") #Comentar si se desea procesar otra fecha
date_to_process="${date:-$date_to_process}"

reduct=$3
############################ PATH VARIABLES ############################
ruta_procdata="/media/soporte/data/"

ruta_especreduced_out="/home/soporte/Desktop/Reducted/BARRANCA/"
graphics_spec_reducted="/home/soporte/Desktop/Reducted/Pictures/BARRANCA/"

ruta_moments_out_r="/home/soporte/Desktop/BARRANCA/MomentsFloyd_Filt/"
ruta_moments_out="/media/soporte/PROCDATA/MomentsFloyd/"
########################################################################
if [[ $reduct == 'r' ]]
then
    ###################### Reduction ########################
    reduct=1
    cd $HOME/Actualizado/Reduccion/
    echo "Starting Reduction!"

    echo "./Reduccion_total.sh" $date_to_process $2 $1 $ruta_procdata $ruta_especreduced_out $graphics_spec_reducted
    ./Reduccion_total.sh $date_to_process $2 $1 $ruta_procdata $ruta_especreduced_out $graphics_spec_reducted
    sleep 5
    #########################################################
    
    ######################## Moments ########################
    cd $HOME/Actualizado/Procesamiento/Plottings/
    echo "Starting Moments"
    echo    "reduct" $reduct
    echo "./Moment.sh" $ruta_especreduced_out $1 $2 $date_to_process $reduct
    ./Moment.sh $ruta_especreduced_out $1 $2 $date_to_process $reduct
    sleep 3
    #Espera a que los nuevos archivos de momentos hayan terminado de generarse
    COUNT=$(screen -list | grep -c "MOMENTS_HF")
    while [  $COUNT != "0" ]
    do
        sleep 2
        echo "Waiting for Moments to finish"
        COUNT=$(screen -list | grep -c "MOMENTS")
    done
    ####################################################################
    
    ############## Plotting Parameters SNR, DOPLLER y OUT ##############
    echo "./Plot_RTDI_OUT.sh" $ruta_moments_r $1 $2 $date_to_process
    ./Plot_RTDI_OUT.sh $ruta_moments_r $1 $2 $date_to_process
    sleep 1
    COUNT=$(screen -list | grep -c "PlottingParam_Floyd")
    while [  $COUNT != "0" ]
    do
	    echo "Waiting for PlottingParam RTDI Out to finish"
	    COUNT=$(screen -list | grep -c "PlottingParam")
    done

else
    reduct=0
    cd $HOME/Actualizado/Procesamiento/Plottings/
    echo "Starting Moments"
    echo "reduct" $reduct
    echo "./Moment.sh" $ruta_procdata $1 $2 $date_to_process $reduct
    ./Moment.sh $ruta_procdata $1 $2 $date_to_process $reduct
    sleep 1
    #Espera a que los nuevos archivos de momentos hayan terminado de generarse
    COUNT=$(screen -list | grep -c "MOMENTS_HF")
    while [  $COUNT != "0" ]
    do
        sleep 3
        echo "Waiting for Moments to finish"
        COUNT=$(screen -list | grep -c "MOMENTS")
    done
    ############## Plotting Parameters SNR, DOPLLER y OUT ############
    echo "./Plot_RTDI_OUT.sh" $ruta_moments $1 $2 $date_to_process
    ./Plot_RTDI_OUT.sh $ruta_moments $1 $2 $date_to_process
    sleep 1
    COUNT=$(screen -list | grep -c "PlottingParam_Floyd")
    while [  $COUNT != "0" ]
    do
	    echo "Waiting for PlottingParam RTDI Out to finish"
	    COUNT=$(screen -list | grep -c "PlottingParam")
    done
fi
############################  
