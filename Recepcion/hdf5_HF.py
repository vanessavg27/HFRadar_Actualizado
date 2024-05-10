import glob
import numpy
import math
import datetime
import time
import pickle
import os
#import astropy.io.ascii
import h5py

debug = False

def hdf_new(dirn, dtype):
    #dtype = numpy.complex64
    files = glob.glob("%s/20*/rf@*.h5"%(dirn))
    files.sort()
    files = files[1:]
    #dtype = numpy.float32 ## mal
    result = {}
    ## Lectura Hdf5
    data = h5py.File(files[0],'r')
    result['file_size'] = len(numpy.array(data['rf_data']))
    result['file_list'] = files
    result['max_n']     = result['file_size']*len(files)
    result['dtype']     = dtype
    result['scale']     = 1
    result['cache']     = numpy.zeros(result['file_size'],dtype=numpy.complex64)
    result['cache_idx'] = -1
    result['t0']        = int(str(numpy.array(data['rf_data_index'])[0,0])[0:10])
    data.close()

    return result


def hdf_read(hdf,idx, length):
#length = 10000 equivale 
    idx = int(idx)
    length = int(length)
    files  = hdf['file_list']
    f0_idx = int(math.floor(idx/hdf['file_size']))
    fn_idx = int(math.floor((idx+length-1)/hdf['file_size']))+1
    res_vec = numpy.zeros([length],dtype=numpy.complex64)
    if debug:
        print("f0",f0_idx," ",fn_idx)
    c0_idx = idx % hdf['file_size']
    c1_idx = (idx+length-1)%hdf['file_size']+1
    if debug:
        print("c0",c0_idx," c1",c1_idx)
    n_read = 0

    for f_idx in range(f0_idx,fn_idx):
        c0 = 0
        c1 = hdf['file_size']
        if f_idx == f0_idx:
            c0 = c0_idx
        if f_idx+1 == fn_idx:
            c1 = c1_idx
        if debug:
            print("Open File, ",f_idx,"\n")

        if hdf['cache_idx'] != f_idx:
            data = h5py.File(files[f_idx],'r')
            a = numpy.array(data['rf_data'],dtype = hdf['dtype'])
            a.resize((hdf['file_size'],))
            data.close()
            
            hdf['cache'] = a
            hdf['cache_idx'] = f_idx
            
            
        
        res_vec[numpy.arange(c1-c0)+n_read] = hdf['cache'][c0:c1]
        #print("RES_VEC: ",res_vec)
        
        n_read = n_read + (c1-c0)

    return res_vec