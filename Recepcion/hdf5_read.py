import glob
import numpy
import math
import datetime
import time
import pickle
import os
import astropy.io.ascii
import h5py
itemsize=8
dtype = numpy.float32


dir = '/media/soporte/RAWDATA/'
doy = 'd2022136_1618'
ch='/0'
dirn = dir+doy+ch
#rutas = glob.glob("%s/%s*.h5"%(dir,doy))

#gdf_0 = [gdf.new_gdf("%s/%s"%(dirn,ch0[i]), dtype = numpy.float32, itemsize = 8) for i in elementos ]
############### SIRVEE DESDE AQUI ####################
files = glob.glob("%s/20*/*.h5"%(dirn))
files.sort()
files = files[1:]
itemsize=8
dtype = numpy.float32

result = {}

data = h5py.File(files[0],'r')

result['file_size'] = len(numpy.array(data['rf_data']))
result['file_list'] = files
result['max_n']     = result['file_size']*len(files)
result['dtype']     = dtype
#result['t0']        = tiempo
data.close()

return result
#result['cache']     = numpy.zeros(result['file_size'],dtype=numpy.int64)

######################## READ_VECT #############################
i = 0 
j = 0
an_len = 100000
clen = 10000
idx_proc = an_len * i
idx0 = idx_proc + j*clen
files = result[]


def read_vec(hdf, idx, length):
    idx = int(idx)
    length = int(length)
    files =
    res_vec = numpy.zeros([length],dtype=numpy.complex64)
    






    pass













return result


