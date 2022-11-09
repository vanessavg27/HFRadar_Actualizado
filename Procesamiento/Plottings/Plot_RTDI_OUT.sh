#!/bin/bash

#./Plot_RTDI_OUT.sh {path_spectra} {location} {mode_operation} {date}
#Normal mode HYO  :    ./Plot_RTDI_OUT.sh "/media/soporte/PROCDATA/MomentsFloyd_Filt/" 21 1 "2022/08/29"
#Campaign mode HYO:    ./Plot_RTDI_OUT.sh "/media/soporte/RAWDATA/HYOa/" 1 21 "2022/08/29"


#Modo normal
if [ $3 = "0" ]
then
	echo "Modo normal"
	echo "HFA0_C0"
	python3 PLOT_SNR_RTDI_OUT.py  -path $1 -f 2.72216796875   -C 0  -code 0 -lo $2 -date $4
	#echo "HFA0_C1"
	python3 PLOT_SNR_RTDI_OUT.py  -path $1 -f 2.7221646392718 -C 0  -code 1 -lo $2 -date $4
	echo "HFA0_C2"
	python PLOT_SNR_RTDI_OUT.py  -path $1 -f 2.7221712982282 -C 0  -code 2 -lo $2 -date $4


	echo "HFA1_C0"
	python3 PLOT_SNR_RTDI_OUT.py  -path $1 -f 3.64990234375   -C 0  -code 0 -lo $2 -date $4
	echo "HFA1_C1"
	python3 PLOT_SNR_RTDI_OUT.py  -path $1 -f 3.6499056732282 -C 0  -code 1 -lo $2 -date $4
	echo "HFA1_C2"
	python3 PLOT_SNR_RTDI_OUT.py  -path $1 -f 3.6498990142718 -C 0  -code 2 -lo $2 -date $4
#Campaña
elif [ $3 = "1" ]
then
	echo "Modo campaña"
	echo "HFA0_C0"
	python3 PLOT_SNR_RTDI_OUT.py  -path $1 -f 2.72216796875   -C 1  -code 0 -lo $2 -date $4
	echo "HFA0_C1"
	python3 PLOT_SNR_RTDI_OUT.py  -path $1 -f 2.7221646392718 -C 1  -code 1 -lo $2 -date $4
	echo "HFA0_C2"
	python3 PLOT_SNR_RTDI_OUT.py  -path $1 -f 2.7221712982282 -C 1  -code 2 -lo $2 -date $4


	echo "HFA1_C0"
	python3 PLOT_SNR_RTDI_OUT.py  -path $1 -f 3.64990234375   -C 1  -code 0 -lo $2 -date $4
	echo "HFA1_C1"
	python3 PLOT_SNR_RTDI_OUT.py  -path $1 -f 3.6499056732282 -C 1  -code 1 -lo $2 -date $4
	echo "HFA1_C2"
	python3 PLOT_SNR_RTDI_OUT.py  -path $1 -f 3.6498990142718 -C 1  -code 2 -lo $2 -date $4
else
	echo "Error en flag de campaña en GenerateMoments.sh"
	exit 1
fi
