#!/usr/bin/env python3
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
import digital_rf as drf
import gr_digital_rf

class read_receiver():

    def __init__(self,op):
        self.op = op
    
    def start(self):
        fg = gr.top_block()

        path_read = self.op.path_read + "d2023304_0845/"
        self.reader= gr_digital_rf.digital_rf_source(path_read,channels=['0','1'],start=[0,0,],
                                                    end=[1999999,1999999,],
                                                    repeat=True,
                                                    throttle=False,
                                                    gapless=False,
                                                    min_chunksize=None,)
        
        middle_frequency = (op.centerfreq[0]+op.centerfreq[1])/2
        desviations_frequencies= self.var_freq(3,op.centerfreq[0],op.centerfreq[1]) # Cantidad de pares de frecuencias 3 

        fg = gr.top_block()
        print("Desviaciones causa:",desviations_frequencies)
        print("Desviacion [0]: ",desviations_frequencies[0])
        print("Desviacion [1]: ",desviations_frequencies[1])
        print("Desviacion [2]: ",desviations_frequencies[2])

        signals_sine =\
                      [analog.sig_source_c(self.op.sample_rate, analog.GR_SIN_WAVE, +1.0*desviations_frequencies[0], 1, 0),
                      analog.sig_source_c(self.op.sample_rate, analog.GR_SIN_WAVE, -1.0*desviations_frequencies[0], 1, 0),
                      analog.sig_source_c(self.op.sample_rate, analog.GR_SIN_WAVE, +1.0*desviations_frequencies[1], 1, 0),
                      analog.sig_source_c(self.op.sample_rate, analog.GR_SIN_WAVE, -1.0*desviations_frequencies[1], 1, 0),
                      analog.sig_source_c(self.op.sample_rate, analog.GR_SIN_WAVE, +1.0*desviations_frequencies[2], 1, 0),
                      analog.sig_source_c(self.op.sample_rate, analog.GR_SIN_WAVE, -1.0*desviations_frequencies[2], 1, 0)]
        print("Signals sine")
        
        mixers = [ blocks.multiply_vcc(1) for i in range(12)]

        taps_low_pass_filter = firdes.low_pass(1, self.op.sample_rate, 50e3, 20e3, firdes.WIN_HAMMING, 6.76)
    	#my_taps = firdes.low_pass(1, samp_rate, 0.2e6, 0.1e6, firdes.WIN_HAMMING, 6.76)
        print("taps_low_pass_filter 200e3:",type(taps_low_pass_filter))
        self.low_pass_filters = [filter.fir_filter_ccf(20, taps_low_pass_filter) for i in range(12)]

        DOY=time.strftime("%Y%j", time.gmtime(time.time()-5*3600))

        dir_name = "%s/d%s"%(op.outputdir,DOY)
        print("DIRECTORIO: ",dir_name)

        self.writer = gr_digital_rf.digital_rf_sink(
            "%s"%(dir_name),
            channels=['0','1','2','3','4','5','6','7','8','9','10','11',],
            dtype=numpy.complex64,
            subdir_cadence_secs=3600,
            file_cadence_millisecs=1000,
            sample_rate_numerator=int(self.op.sample_rate),
            sample_rate_denominator=1,
            start=None,
            ignore_tags=False,
            is_complex=True,
            num_subchannels=1,
            uuid_str=None,
            center_frequencies=(
                None
            ),
            metadata={},
            is_continuous=True,
            compression_level=0,
            checksum=False,
            marching_periods=True,
            stop_on_skipped=False,
            stop_on_time_tag=False,
            debug=False,
            min_chunksize=None,
        )

        ##################### FREQs BASE #################### 
        ################## CH0 - DESV base ##################
                        ########## 2.72 MHz #########
        fg.connect((self.reader,0), (mixers[0],0))
        fg.connect((signals_sine[0], 0), (mixers[0], 1))
        fg.connect((mixers[0],0), (self.low_pass_filters[0],0))
        fg.connect((self.low_pass_filters[0],0), (self.writer,0))

                        ########## 3.64 MHz #########
        fg.connect((self.reader,0), (mixers[1],0))
        fg.connect((signals_sine[1], 0), (mixers[1], 1))
        fg.connect((mixers[1],0), (self.low_pass_filters[1],0))
        fg.connect((self.low_pass_filters[1],0), (self.writer,1))

        ################## CH1 - DESV base ##################
                        ########## 2.72 MHz #########
        fg.connect((self.reader,1), (mixers[2],0))
        fg.connect((signals_sine[0], 0), (mixers[2], 1))
        fg.connect((mixers[2],0), (self.low_pass_filters[2],0))
        fg.connect((self.low_pass_filters[2],0), (self.writer,2))

                        ########## 3.64 MHz #########
        fg.connect((self.reader,1), (mixers[3],0))
        fg.connect((signals_sine[1], 0), (mixers[3], 1))
        fg.connect((mixers[3],0), (self.low_pass_filters[3],0))
        fg.connect((self.low_pass_filters[3],0), (self.writer,3))

        ##################### FREQ +alejadas ################
        ################## CH0 - DESV base + 3.329 ##########
                        ########## 2.72 MHz #########
        fg.connect((self.reader,0), (mixers[4],0))
        fg.connect((signals_sine[2], 0), (mixers[4], 1))
        fg.connect((mixers[4],0), (self.low_pass_filters[4],0))
        fg.connect((self.low_pass_filters[4],0), (self.writer,4))

                        ########## 3.64 MHz #########
        fg.connect((self.reader,0), (mixers[5],0))
        fg.connect((signals_sine[3], 0), (mixers[5], 1))
        fg.connect((mixers[5],0), (self.low_pass_filters[5],0))
        fg.connect((self.low_pass_filters[5],0), (self.writer,5))

        ################# CH1 - desv base +3.329 ###########
        fg.connect((self.reader,1), (mixers[6],0))
        fg.connect((signals_sine[2], 0), (mixers[6], 1))
        fg.connect((mixers[6],0), (self.low_pass_filters[6],0))
        fg.connect((self.low_pass_filters[6],0), (self.writer,6))

        fg.connect((self.reader,1), (mixers[7],0))
        fg.connect((signals_sine[3], 0), (mixers[7], 1))
        fg.connect((mixers[7],0), (self.low_pass_filters[7],0))
        fg.connect((self.low_pass_filters[7],0), (self.writer,7))


        ##################### FREQ +cercanas #################
        ################## CH0 - DESV base - 3.329 ##########
        fg.connect((self.reader,0), (mixers[8],0))
        fg.connect((signals_sine[4], 0), (mixers[8], 1))
        fg.connect((mixers[8],0), (self.low_pass_filters[8],0))
        fg.connect((self.low_pass_filters[8],0), (self.writer,8))


        fg.connect((self.reader,0), (mixers[9],0))
        fg.connect((signals_sine[5], 0), (mixers[9], 1))
        fg.connect((mixers[9],0), (self.low_pass_filters[9],0))
        fg.connect((self.low_pass_filters[9],0), (self.writer,9))

        ################# CH1 - desv base +3.329 ###########
        fg.connect((self.reader,1), (mixers[10],0))
        fg.connect((signals_sine[4], 0), (mixers[10], 1))
        fg.connect((mixers[10],0), (self.low_pass_filters[10],0))
        fg.connect((self.low_pass_filters[10],0), (self.writer,10))

        fg.connect((self.reader,1), (mixers[11],0))
        fg.connect((signals_sine[5], 0), (mixers[11], 1))
        fg.connect((mixers[11],0), (self.low_pass_filters[11],0))
        fg.connect((self.low_pass_filters[11],0), (self.writer,11))

        fg.start()

    def var_freq(self,cantidad,f0,f1):
        paso = 100000000/pow(2,32)*143   # Variacion de 3.3294... Hz
        i = 0                            # Contador para frecuencias
        desv = []
        while(i<cantidad):
            print("Contador desviacion",i)
            FREQ =[f0+((-1)**i)*math.ceil(i/2)*paso,f1-((-1)**i)*math.ceil(i/2)*paso]
            desv.append((FREQ[1]-FREQ[0])/2)
            i+=1
            print("   ",FREQ)
        return desv 
'''ruta="/home/soporte/DATA/d2023304/"
    ruta1="/home/soporte/DATA/d2023304/0/"
lectura = gr_digital_rf.digital_rf_channel_source(ruta,channels=['0', '1'],
                                        start=None,end=None,
                                        repeat=False,throttle=False,
                                        gapless=False,min_chunksize=None)

    

#print(canal.out_sig())
#print(canal._properties["samples_per_second"])
#print("LECTURA",lectura)

    lectura = gr_digital_rf.digital_rf_channel_source(ruta,channels=['0', '1'],
                                        start=None,end=None,
                                        repeat=False,throttle=False,
                                        gapless=False,min_chunksize=None)
'''
                          # Nuevas desviaciones con variacion de +- 3.329 ... Hz

if __name__ == '__main__':

    """
    Version 1.0.2.1_beta_ric

    Adquisition

    python ./hfrx2_ric_new.py -a 192.168.10.2 -r 2000000 -y 2 -d 10
    -b external -o /full_path/rawdata/ -e 11.5 -c 2.72e6,3.64e6

    """

    print("######### PROGRAMA DE RECEPCION HF ###########")
    parser = OptionParser(option_class=eng_option, usage="%prog: [options]")

    parser.add_option("-r", "--samplerate", dest="sample_rate", type="int",action="store",
                      default=2000000,
                      help="Sample rate (default = 1 MHz).")

    parser.add_option("-t", "--rep", dest="rep", type="int",action="store", default=1,
                      help="Repetition time (default = 1 s)")

    parser.add_option("-d", "--dec", dest="dec", type="int",action="store", default=10,
                      help="Integrate and decimate by this factor (default = 10)")

    parser.add_option("-z", "--filesize", dest="filesize", type="int",action="store",
                      help="File size (samples 1 second)")

    parser.add_option("-s", "--starttime", dest="start_time", type="int", action="store",
                      help="Start time (unix seconds)")
    ################################ NUEVAS FRECUENCIAS BASE ###########################################
    #parser.add_option("-c", "--centerfreq",dest="centerfreq", action="store", type="string",default="2.728271484375e6,3.643798828125e6",
    #                  help="Center frequency (default 2.4e9,2.4e9)")
    
    ################################## FRECUENCIAS BASE ORIGINALES #######################################
    parser.add_option("-c", "--centerfreq",dest="centerfreq", action="store", type="string",default="2.72216796875e6,3.64990234375e6",
                      help="Center frequency (default 2.4e9,2.4e9)")

    parser.add_option("-b", "--clocksource",dest="clocksource", action="store", type="string", default="external",
                      help="Clock source (default gpsdo)")
    
    parser.add_option("-p", "--path_r",dest="path_read", action="store", type="string", default="/home/soporte/DATA/",
                      help="Output destination (default hfrx-datestring)")

    parser.add_option("-o", "--outputdir",dest="outputdir", action="store", type="string", default="/home/soporte/DATA/MULTIFREQ",
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
        op.filesize = op.sample_rate/(op.dec)

    s = read_receiver(op)
    s.start()