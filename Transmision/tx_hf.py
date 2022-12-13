#!/usr/bin/env python3
#
# Beacon transmit
# Options:
# - center frequency
# - start time
# - transmit file
# gnuradio 3.6.4.1
# Timing format: (start_time, repetition_time). For example (0,100) would indicate that the experiment repeats
# every 100 s and start 0 seconds past midnight.
#

from gnuradio import eng_notation
from gnuradio import analog
from gnuradio import gr
from gnuradio import uhd
from gnuradio import blocks
from gnuradio.eng_option import eng_option
#from gnuradio.gr import firdes
from optparse import OptionParser
import math, time, calendar
import sampler_util
import numpy

#./tx_hf.py -f 2.72216796875 -c 0 -z 0

class beacon_transmit:
    def __init__(self,args):
        self.args = args

    def print_info(self,uinfo):
        print("    Mboard id: %s"%uinfo.get("mboard_id"))
        print("    Mboard serial %s"%uinfo.get("mboard_serial"))
        print("    Tx db name %s"%uinfo.get("tx_subdev_name"))

    def start(self):
        txlog = open("hftx.log","w")
        tb = gr.top_block()
        sampler_util.real_time_scheduling()
        tstart_tx = time.time()
        tnow = time.time()
        dev_str = "addr=%s,send_buff_size=1000000"%(self.args.ip)
        sink = uhd.usrp_sink(
            device_addr=dev_str,
            stream_args=uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                channels=range(1),
            ),
        )

        #async_msgq = gr.msg_queue(0)Commented by Alejandro
        #async_src = uhd.amsg_source("", async_msgq)Commented by Alejandro
        #        sink_msgsrc = uhd.amsg_source(uhd.device_addr(self.args.ip),sink_queue)

        #sink.set_clock_source(self.args.clocksource, 0)
        sink.set_time_source(self.args.clocksource, 0)
        print (sink.get_mboard_sensor("ref_locked"))

        next_time = sampler_util.find_next(self.args.start_time, per=self.args.rep)
        print("Starting at ",next_time)

        tt = time.time()
        while tt-math.floor(tt) < 0.3 or tt-math.floor(tt) > 0.5:
            tt = time.time()
            time.sleep(0.01)

        sink.set_time_unknown_pps(uhd.time_spec(math.ceil(tt)+1.0))
        sink.set_start_time(uhd.time_spec(next_time + self.args.clockoffset/1e6))

        sink.set_samp_rate(self.args.sample_rate)
        sink.set_center_freq(self.args.center_freq, 0)
        print("Actual center freq %1.8f Hz"%(sink.get_center_freq(0)))
        print("    Gain: %s"%(self.args.gain))
        print("    Code file:",self.args.codefile)
        sink.set_gain(self.args.gain, 0)
        sink.set_antenna(self.args.txport, 0)
        #code_source = gr.file_source(gr.sizeof_gr_complex*1, self.args.codefile, True)


        code_vector = numpy.fromfile(self.args.codefile,dtype=numpy.complex64)
        code_source = blocks.vector_source_c(code_vector.tolist(), True)
        #multiply = gr.multiply_const_vcc((0.5, ))
        print("    Amplitud: %s"%(self.args.amplitude))
        multiply = blocks.multiply_const_vcc((self.args.amplitude, ))

        tb.connect(code_source, multiply, sink)
        #        tb.connect((async_src, msg), (sink_queue, 0))
        tb.start()
        self.print_info(sink.get_usrp_info(0))
        print("Starting")
        print("Restart time: "+str(self.args.restart_time))
        while(True):
            tnow = time.time()
            if (tnow - tstart_tx) > self.args.restart_time:
                tb.stop()
                exit(0)
            print (sink.get_mboard_sensor("ref_locked"))
            if self.args.clocksource == "gpsdo":
                txlog.write("%s %s\n"%(sampler_util.time_stamp(),sink.get_mboard_sensor("gps_locked")))


            txlog.write("%s %s\n"%(sampler_util.time_stamp(),sink.get_mboard_sensor("ref_locked")))
            txlog.write("%s\n"%(sampler_util.time_stamp()))

            time.sleep(10.0)


if __name__ == '__main__':
    parser = OptionParser(option_class=eng_option, usage="%prog: [options]")

    parser.add_option("-a", "--address", dest="ip", type=str,action="store", default="192.168.20.8",help='Device address (ip number).')

    parser.add_option("-r", "--samplerate", dest="sample_rate", type=int,action="store", default=1000000,help="Sample rate (Hz).")

    parser.add_option("-t", "--rep", dest="rep", type=int,action="store",default=1,help="Repetition time (s)")

    parser.add_option("-s", "--starttime", dest="start_time", type=int, action="store",help="Start time (s)")

    parser.add_option("-x", "--txport", dest="txport", type=str, action="store", default="TX/RX",help="TX port")

    parser.add_option("-g", "--gain", dest="gain", type=float, action="store", default=0.0,help="Transmit gain (default 0.0 dB)")

    parser.add_option("-f", "--centerfreq",dest="center_freq", action="store", type=float, default=3.64990234375e6,help="Center frequency (default 1.9e6)")

    parser.add_option("-c", "--codefile",dest="codefile", action="store", type=str,help="Transmit code file.")

    parser.add_option("-b", "--clocksource",dest="clocksource", action="store", type=str, default="external",help="Clock source (default gpsdo).")

    parser.add_option("-o", "--clockoffset",dest="clockoffset", action="store", type=float, default=0.0,help="Clock offset in microseconds (default 0.0 us).")

    parser.add_option("-z", "--restart",dest="restart_time", action="store", type=float, default=3600.0,help="Restart every n seconds, to realign clock (default 3600 s).")

    parser.add_option("-p", "--amplitude", dest="amplitude", type=float, action="store", default=0.5, help="Amplitude factor [0,1] (default 0.25)")

    (op,args) = parser.parse_args()

    #self.args.ip = "192.168.10.2"
    op.recv_buff_size = 100000
    op.send_buff_size = 100000

    if op.start_time == None:
        op.start_time = math.ceil(time.time())

    if op.codefile == None:
        op.codefile = "code-000000.bin"
    op.amp=1.0
    tx = beacon_transmit(op)
    tx.start()
