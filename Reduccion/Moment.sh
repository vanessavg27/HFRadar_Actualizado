#/bin/bash

#./Moment.sh {path_spectra} {mode_operation} {location} {date}
#Normal mode HYO  :    ./Moment.sh "/media/soporte/RAWDATA/HYOa/" 0 21 "2022/08/20"
#Campaign mode HYO:    ./Moment.sh "/media/soporte/RAWDATA/HYOa/" 1 21 "2022/08/29"

#Modo normal
if [ $2 = "0" ]
then
	echo "Modo normal"
#Campaña
elif [ $2 = "1" ]
then
	echo "Modo campaña"
fi

	echo "HFA0_C0"
	python3 Moment.py  -path $1 -f 2.72216796875   -C $2  -code 0 -lo $3 -date $4 -R 1
	echo "HFA0_C1"
	python3 Moment.py  -path $1 -f 2.7221646392718 -C $2  -code 1 -lo $3 -date $4 -R 1
	echo "HFA0_C2"
	python3 Moment.py  -path $1 -f 2.7221712982282 -C $2  -code 2 -lo $3 -date $4 -R 1


	echo "HFA1_C0"
	python3 Moment.py  -path $1 -f 3.64990234375   -C $2  -code 0 -lo $3 -date $4 -R 1
	echo "HFA1_C1"
	python3 Moment.py  -path $1 -f 3.6499056732282 -C $2  -code 1 -lo $3 -date $4 -R 1
	echo "HFA1_C2"
	python3 Moment.py  -path $1 -f 3.6498990142718 -C $2  -code 2 -lo $3 -date $4 -R 1

