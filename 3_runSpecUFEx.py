#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:02:51 2021

@author: theresasawi
"""

import h5py
import numpy as np
import sys

import pandas as pd
from matplotlib import pyplot as plt

sys.path.append('functions/')
from setParams import setParams
# from generators import gen_sgram_QC

import tables
tables.file._open_files.close_all()
from specufex import BayesianNonparametricNMF, BayesianHMM


#%% load project variables: names and paths

key = sys.argv[1]

# key = 'Parkfield_Repeaters'




#%%
### do not change these ###

pathProj, pathCat, pathWF, network, station, channel, channel_ID, filetype, cat_columns = setParams(key)



dataH5_name = f'data_{key}.hdf5'
dataH5_path = pathProj + '/H5files/' + dataH5_name
SpecUFEx_H5_name = f'SpecUFEx_{key}.hdf5'
SpecUFEx_H5_path = pathProj + '/H5files/' + SpecUFEx_H5_name
sgramMatOut = pathProj + 'matSgrams/'## for testing
pathWf_cat  = pathProj + 'wf_cat_out.csv'
pathSgram_cat = pathProj + f'sgram_cat_out_{key}.csv'

sgram_cat = pd.read_csv(pathSgram_cat)



#%% get spectrograms from H5


X= []
H5=True

if H5:
    with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:
        for evID in fileLoad['spectrograms']:
            specMat = fileLoad['spectrograms'].get(evID)[:]
            Xis = specMat
            X.append(Xis)

X = np.array(X)

#%%
nmf = BayesianNonparametricNMF(X.shape)
nmf.fit(X, verbose=1)
Vs = nmf.transform(X)

#%%

hmm = BayesianHMM(nmf.num_pat, nmf.gain)
hmm.fit(Vs)
fingerprints, As, gams = hmm.transform(Vs)

#%%
plt.imshow(fingerprints[0])
#%%
#%%
#%%

# =============================================================================
# save output to H5
# =============================================================================
with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:


    ##fingerprints are top folder
    if 'fingerprints' in fileLoad.keys():
        del fileLoad["fingerprints"]

    fp_group = fileLoad.create_group('fingerprints')

    out_group               = fileLoad.create_group("SpecUFEX_output")

    ACM_group               = fileLoad.create_group("SpecUFEX_output/ACM")
    STM_group               = fileLoad.create_group("SpecUFEX_output/STM")
    gain_group              = fileLoad.create_group("SpecUFEX_output/ACM_gain")



    for i, evID in enumerate(fileLoad['spectrograms']):
        fp_group.create_dataset(name= evID, data=fingerprints[i])
        # ACM_group.create_dataset(name=evID,data=As[i]) #ACM
        # STM_group.create_dataset(name=evID,data=gam[i]) #STM


    W_group                      = fileLoad.create_group("SpecUFEX_output/W")
    gain_group                   = fileLoad.create_group("SpecUFEX_output/gain")
    EB_group                     = fileLoad.create_group("SpecUFEX_output/EB")
    RMM_group                    = fileLoad.create_group("SpecUFEX_output/RMM")



    W_group.create_dataset(name='W',data=nmf.EW)
    EB_group.create_dataset(name=evID,data=hmm.EB)
    # RMM_group.create_dataset(name=evID,data=RMM)
    gain_group.create_dataset(name='gain',data=nmf.gain) #same for all data

#%%
#%%



#%%



#%%
