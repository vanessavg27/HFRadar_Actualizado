#/bin/bash

#./Moment.sh {path_spectra} {mode_operation} {location} {date}
#Normal   mode HYO:    ./Moment.sh "/media/soporte/RAWDATA/HYOa/" 21 0  "2022/08/20" 1
#Campaign mode HYO:    ./Moment.sh "/media/soporte/RAWDATA/" 21 1 2022/08/29" 1
reducted=$5
path_o="/home/soporte/Desktop/BARRANCA/MomentsFloyd/"
#Modo normal
if [ $2 = "0" ]
then
	echo "Modo Normal"
#Campaña
elif [ $2 = "1" ]
then
	echo "Modo Campaña"
fi

	echo "HFA0_C0"
	python3 Moment.py  -path $1 -f 2.72216796875   -lo $2 -C $3  -code 0  -date $4 -R $reducted -path_o $path_o
	echo "HFA0_C1"
	python3 Moment.py  -path $1 -f 2.7221646392718 -lo $2 -C $3  -code 1  -date $4 -R $reducted -path_o $path_o
	echo "HFA0_C2"
	python3 Moment.py  -path $1 -f 2.7221712982282 -lo $2 -C $3  -code 2  -date $4 -R $reducted -path_o $path_o


	echo "HFA1_C0"
	python3 Moment.py  -path $1 -f 3.64990234375   -lo $2 -C $3  -code 0  -date $4 -R $reducted -path_o $path_o
	echo "HFA1_C1"
	python3 Moment.py  -path $1 -f 3.6499056732282 -lo $2 -C $3  -code 1 -date $4 -R $reducted  -path_o $path_o
	echo "HFA1_C2"
	python3 Moment.py  -path $1 -f 3.6498990142718 -lo $2 -C $3  -code 2 -date $4 -R $reducted  -path_o $path_o
