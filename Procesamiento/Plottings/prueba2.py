#!/usr/bin/env python3.6

filename = "/home/igp-114/Pictures/GRAPHICS_FLOYD_JRO_B/sp21_f0/d2022197/"
code = 2
freq = 2.72
dates = 
if freq < 3.6:
    ngraph = 0
else:
    ngraph = 1

from datetime import datetime
day = datetime.strptime(dates, "%Y/%m/%d")
Days = day.strftime("%Y%j")

location_dict = {11:"JRO_A", 12: "JRO_B", 21:"HYO_A", 22:"HYO_B", 31:"MALA",
				41:"MERCED", 51:"BARRANCA", 61:"OROYA"}

identifier = 'sp%s1_f%s'%(code, ngraph)
Channels = ['0','1']

for channel in Channels:
    print("Iniciando con canal: ",channel)
    file= str('M%s'%())