#!/usr/bin/env python3
#
# HF radar receiver
#
# The synchronization is done using a reference 1 PPS and 10 MHz signal.
# We assume that the code has a certain cycle length after it repeats, thus
# offering a logical place to synchronize to. The user gives the starting time
# in unix time and specifies the repeat time of the code. The receiver will then
# start sampling at the beginning of the next cycle.
#
# Tune to a certain frequency and store data as files of a certain length.
# Use a high over-the-wire sample rate and perform additional integration
# and decimation in single precision floating point format to achieve
# higher dynamic range, often needed for continuous radar.
#
# (c) 2013 Juha Vierinen
# Multiple frequencies 2022 Vanessa Vasquez
 
from __future__ import division   #Libreria para obtener correctas divisiones
from gnuradio import eng_notation
from gnuradio import gr
#from gnuradio import blks2
from gnuradio import uhd
from gnuradio import filter
from gnuradio.eng_option import eng_option
from optparse import OptionParser 
from gnuradio.filter import firdes
from gnuradio import analog  # Libreria para modulacion generador de funciones analogicas



import sys, time, os, math, re, calendar, glob

import sampler_util
from gnuradio import blocks
#import filesink2
import stuffr
import numpy
import shutil

import gr_digital_rf

class beacon_receiver:

    def __init__(self,op):
        self.op = op  # Libreria OptParse[ se ingresan varias variables en una 
                      # sola linea en este atributo]:

    def start(self):

        u = uhd.usrp_source(
            device_addr="addr=%s"%(op.ip),
            stream_args=uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                channels=range(2),
                ),
            )

        u.set_subdev_spec(op.subdev) 
        #  u.set_subdev_spec(op.subdev[1],0)
        u.set_samp_rate(op.sample_rate)

        #u.set_center_freq(op.centerfreq[0], 0)
        #u.set_center_freq(op.centerfreq[1], 1)
        middle_frequency = (op.centerfreq[0]+op.centerfreq[1])/2
        u.set_center_freq(middle_frequency, 0)
        u.set_center_freq(middle_frequency, 1)

        u.set_gain(op.gain,0)
        u.set_gain(op.gain,1)
        print("Actual center freq %1.6f Hz %1.6f"%(u.get_center_freq(0),u.get_center_freq(1)))
        ######### CLOCK SOURCE #########

        if op.clocksource != "none":
            u.set_clock_source(op.clocksource, 0)
            u.set_time_source(op.clocksource, 0)
            print(u.get_mboard_sensor("ref_locked"))
        if op.clocksource == "gpsdo":
            print(u.get_mboard_sensor("gps_time"))
            print(u.get_mboard_sensor("gps_locked"))
            print( "MOSTRANDO GPS-DO"   ) #GPS - DO

        fg = gr.top_block()

        """ Detecting current DOY """
        DOY=time.strftime("%Y%j", time.gmtime(time.time()-5*3600))

        dir_name = "%s/d%s"%(op.outputdir,DOY)
        print("DIRECTORIO: ",dir_name)
        #If the folder exists add the current hour and minute to "dir_name"
        if os.path.exists(dir_name):
            HHMM=time.strftime("%H%M", time.gmtime(time.time()-5*3600))
            dir_name = "%s/d%s_%s"%(op.outputdir,DOY,HHMM)

        rawdata_parent_folder = op.outputdir
        
        self.check_rawdata_number_of_file(rawdata_parent_folder)
        self.check_rawdata_end_flag(rawdata_parent_folder)

        self.create_structure(dir_name)

        ########## CREACION OBJETO - DIGITAL READER // BLOQUE DE ESCRITURA ##########
        self.digital_rf = \
                gr_digital_rf.digital_rf_sink(
                    "%s"%(dir_name),
                    channels=['0', '1'],
                    dtype=numpy.complex64,
                    #dtype=numpy.int16,
                    subdir_cadence_secs=86400,
                    file_cadence_millisecs=1000,
                    sample_rate_numerator=int(op.sample_rate),
                    #sammple_rate_denominator= op.dec*op.dec0,
                    sample_rate_denominator= 1,
                    start='', ignore_tags=False,
                    is_complex=True, num_subchannels=1,
                    uuid_str=None if ''=='' else '',
                    center_frequencies=None if () is () else (),
                    metadata={},
                    is_continuous=True, compression_level=0,
                    checksum=False, marching_periods=True,
                    stop_on_skipped=False, stop_on_time_tag=False,
                    debug=False,
                    min_chunksize=None if 0==0 else 0,
                )
        
        next_time = sampler_util.find_next(op.start_time, per=op.rep)
        print("Starting at ",next_time)
        print("Starting at ",stuffr.unix2datestr(next_time))
        u.set_start_time(uhd.time_spec(next_time+op.clockoffset/1e6))
        
        fg.connect((u,0), (self.digital_rf,0))
        fg.connect((u,1), (self.digital_rf,1))

        tt = time.time()
        while tt-math.floor(tt) < 0.2 or tt-math.floor(tt) > 0.3:
            tt = time.time()
            time.sleep(0.01)

        if op.clocksource != "none":
            u.set_time_unknown_pps(uhd.time_spec(math.ceil(tt)+1.0))
        
        fg.start()

        #i = 0
        isConfig = False
        while True:
        #while i < 16:
            #i += 1
            if isConfig== False:
               tmp = u.get_mboard_sensor("ref_locked")
			
               if not(tmp.to_bool()):
                  print('NoLocked')
                  fg.stop()
                  exit(0)
                  
            print(tmp)
            isConfig= True
        #print op.runtime

            if op.runtime != -1:
                if time.time()-next_time > op.runtime:
                    print("PARADA")
                    fg.stop()
                    exit(0)






    def create_structure(self, new_dir_name):

        os.system("mkdir -p %s"%(new_dir_name))
        os.system("mkdir -p %s/0"%(new_dir_name))
        os.system("mkdir -p %s/1"%(new_dir_name))


    #create rawdata end flag file
    def create_rawdata_end_flag(self, dir_name):

        os.system("touch %s/rawdata_end_flag"%(dir_name))


    def check_rawdata_end_flag(self, rawdata_parent_folder):
        #Verifica si la generacion de datos crudos finalizo
        #antes de crear la carpeta de adquisicion actual.
        dirns = []
        dirns = dirns + glob.glob("%s/d*"%(rawdata_parent_folder))
        dirns.sort()

        for tmp in dirns[-5:]:
            print("Checking if rawdata_end_flag exists in folder %s"%(tmp))
            if not os.path.isfile("%s/rawdata_end_flag"%(tmp)):
                print("Creating rawdata_end_flag")
                self.create_rawdata_end_flag(tmp)

    def check_rawdata_number_of_file(self,rawdata_parent_folder):
	    #-1 borra lo anterior considera la lista de dias y evalua todo menos el ultimo
        # borra los archivos gdf de cada folder, empezando por los creados de forma creciente y menos el ultimo
        dirns=[]
        dirns = dirns + glob.glob("%s/d*"%(rawdata_parent_folder))
        dirns.sort()
        print("")
        print("****CHECK RAWDATA: Checkeando numeros de files por segundo****")

        for tmp in dirns:#antes estaba dirns[-3:              #Solo analiza el folder 0 y
            #print("TMP",tmp)
            if os.path.isdir(tmp+"/0"):
                print("Analizando cantidad de datos suficientes en: ", tmp)
                number_gdf = glob.glob("%s/%s/20*/rf*.h5"%(tmp,0))
                if len(number_gdf) < 60:
                    print("ELIMINANDO DIRECTORIO",tmp)
                    shutil.rmtree(tmp)
            else:
                continue
        del dirns
        print("")






if __name__ == '__main__':

    """
    Version 1.0.2.1_beta_ric

    Adquisition

    python ./hfrx2_ric_new.py -a 192.168.10.2 -r 2000000 -y 2 -d 10
    -b external -o /full_path/rawdata/ -e 11.5 -c 2.72e6,3.64e6

    """

    print("######### PROGRAMA DE RECEPCION HF ###########")
    parser = OptionParser(option_class=eng_option, usage="%prog: [options]")

    parser.add_option("-a", "--address",
                      dest="ip",
                      type="string",
                      action="store",
                      default="192.168.10.2",#CAMBIAR IP
                      help="Device address (ip number).")

    parser.add_option("-r", "--samplerate", dest="sample_rate", type="int",action="store",
                      default=2000000,
                      help="Sample rate (default = 1 MHz).")

    parser.add_option("-t", "--rep", dest="rep", type="int",action="store", default=1,
                      help="Repetition time (default = 1 s)")

    parser.add_option("-d", "--dec", dest="dec", type="int",action="store", default=10,
                      help="Integrate and decimate by this factor (default = 10)")

    parser.add_option("-y", "--dec0", dest="dec0", type="int",action="store", default=2,
                      help="First decimator factor (default = 2)")

    parser.add_option("-z", "--filesize", dest="filesize", type="int",action="store",
                      help="File size (samples 1 second)")

    parser.add_option("-s", "--starttime", dest="start_time", type="int", action="store",
                      help="Start time (unix seconds)")

    parser.add_option("-f", "--runtime", dest="runtime", type="int", action="store", default=-1,
                      help="Number of seconds to run (seconds)")

    parser.add_option("-x", "--subdev", dest="subdev", type="string", action="store", default="A:A A:B",
                      help="RX subdevice spec (default=RX2)")

      ################################ NUEVAS FRECUENCIAS BASE ###########################################
    #parser.add_option("-c", "--centerfreq",dest="centerfreq", action="store", type="string",default="2.728271484375e6,3.643798828125e6",
    #                  help="Center frequency (default 2.4e9,2.4e9)")
    
    ################################## FRECUENCIAS BASE ORIGINALES #######################################
    parser.add_option("-c", "--centerfreq",dest="centerfreq", action="store", type="string",default="2.72216796875e6,3.64990234375e6",
                      help="Center frequency (default 2.4e9,2.4e9)")


    parser.add_option("-b", "--clocksource",dest="clocksource", action="store", type="string", default="external",
                      help="Clock source (default gpsdo)")

    parser.add_option("-o", "--outputdir",dest="outputdir", action="store", type="string", default="/media/soporte/RAWDATA/",
                      help="Output destination (default hfrx-datestring)")

    parser.add_option("-g", "--gain",dest="gain", action="store", type="float", default=20.0,
                      help="Gain (default 20 dB)")

    parser.add_option("-e", "--clockoffset",dest="clockoffset", action="store", type="float", default=11.5,
                      help="Clock offset in microseconds (default 0.0 us).")

    (op, args) = parser.parse_args()

    op.recv_buff_size = 100000000
    op.send_buff_size = 100000

    if op.start_time == None:
        op.start_time = math.ceil(time.time())

    cf = op.centerfreq.split(",")
    op.centerfreq = [float(cf[0]),float(cf[1])]
    print(op.centerfreq)

    if op.filesize == None:
        op.filesize = op.sample_rate/(op.dec*op.dec0)

    s = beacon_receiver(op)
    s.start()
