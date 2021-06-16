#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:00:19 2021
@author: theresasawi
"""

import h5py
import numpy as np
import sys
import os
import pandas as pd
import tables
import yaml
tables.file._open_files.close_all()
from functions.generators import gen_sgram_QC

# load project variables: names and paths
config_filename = sys.argv[1]
with open(config_filename, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

pathCatWF = config['pathCat']
dataH5_name = f'data_{config["name"]}.hdf5'
dataH5_path = config['pathProj'] + 'H5files/' + dataH5_name
SpecUFEx_H5_name = f'SpecUFEx_{config["name"]}.hdf5'
SpecUFEx_H5_path = config['pathProj'] + 'H5files/' + SpecUFEx_H5_name

# ## for testing
# sgramMatOut = pathProj + 'matSgrams/'

pathWf_cat  = config['pathProj'] + 'wf_cat_out.csv'
pathSgram_cat = config['pathProj'] + f'sgram_cat_out_{config["name"]}.csv'
# get wf catalog
wf_cat = pd.read_csv(pathWf_cat)
evID_list = list(wf_cat.event_ID)



# get sgram params
fmin = config['fmin']
fmax = config['fmax']
winLen_Sec = config['winLen_Sec']
fracOverlap = config['fracOverlap']
nfft = config['nfft']

with h5py.File(dataH5_path,'r+') as fileLoad:
    # sampling rate, Hz
    fs = fileLoad[f"{config['station']}/processing_info"].get('sampling_rate_Hz')[()]
    # number of datapoints
    lenData = fileLoad[f"{config['station']}/processing_info"].get('lenData')[()]

# spectrogram parameters
# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
nperseg = int(winLen_Sec*fs) #datapoints per window segment
noverlap = nperseg*fracOverlap  #fraction of window overlapped

#padding must be longer than n per window segment
if nfft < nperseg:
    nfft = nperseg*2
    print("nfft too short; changing to ", nfft)

mode='magnitude'
scaling='spectrum'
# set args for generator
args = {'station':config['station'],
        'channel':config['channel'],
        'fs': fs,
        'lenData': lenData,
        'nperseg': nperseg,
        'noverlap': noverlap,
        'nfft': nfft,
        'mode': mode,
        'scaling': scaling,
        'fmin': fmin,
        'fmax': fmax}

# put sgrams in h5

### ### ### CREATE GENERATOR ### ### ###
gen_sgram = gen_sgram_QC(config['name'],
                        evID_list=evID_list,
                        dataH5_path = dataH5_path,
                        h5File=fileLoad, #h5 data file
                        trim=True, #trim to min and max freq
                        saveMat=False, #set true to save folder of .mat files
                        sgramOutfile='.', #path to save .mat files
                        **args
                        ) #path to save sgram figures

evID_list_QC_sgram = []

with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:
    n=0
    Nkept=0
    if 'spectrograms' in fileLoad.keys():
        del fileLoad["spectrograms"]
    if 'sgram_normConst' in fileLoad.keys():
        del fileLoad["sgram_normConst"]
    spectrograms_group     = fileLoad.create_group(f"spectrograms")
    sgram_normConst_group  = fileLoad.create_group(f"sgram_normConst")

    while n <= len(evID_list): ## not sure a better way to execute this? But it works
        try:   #catch generator "stop iteration" error
            evID,sgram,fSTFT,tSTFT, normConstant, Nkept,evID_BADones, i = next(gen_sgram) #next() command updates generator
            n = i+1
            evID = str(evID)
            if not evID in spectrograms_group:
                spectrograms_group.create_dataset(name= evID, data=sgram)
                evID_list_QC_sgram.append(evID)
            if not evID in sgram_normConst_group:
                sgram_normConst_group.create_dataset(name= evID, data=normConstant)
        except StopIteration: #handle generator error
            break

    print('N events in evID_list_QC_sgram:', len(evID_list_QC_sgram))
    print('N events in evID_BADones:', len(evID_BADones))

    if 'spec_parameters' in fileLoad.keys():
        del fileLoad["spec_parameters"]

    spec_parameters_group  = fileLoad.create_group(f"spec_parameters")
    spec_parameters_group.clear()
    spec_parameters_group.create_dataset(name= 'fs', data=fs)
    spec_parameters_group.create_dataset(name= 'lenData', data=lenData)
    spec_parameters_group.create_dataset(name= 'nperseg', data=nperseg)
    spec_parameters_group.create_dataset(name= 'noverlap', data=noverlap)
    spec_parameters_group.create_dataset(name= 'nfft', data=nfft)
    spec_parameters_group.create_dataset(name= 'mode', data=mode)
    spec_parameters_group.create_dataset(name= 'scaling', data=scaling)
    spec_parameters_group.create_dataset(name= 'fmin', data=fmin)
    spec_parameters_group.create_dataset(name= 'fmax', data=fmax)
    spec_parameters_group.create_dataset(name= 'fSTFT', data=fSTFT)
    spec_parameters_group.create_dataset(name= 'tSTFT', data=tSTFT)

# merge catalogs
cat_keep_sgram = wf_cat[wf_cat['event_ID'].isin(evID_list_QC_sgram)]

try:
    cat_keep_sgram = cat_keep_sgram.drop(['Unnamed: 0'],axis=1)

except:
    pass

if os.path.exists(pathSgram_cat):
    os.remove(pathSgram_cat)
cat_keep_sgram.to_csv(pathSgram_cat)

# save local catalog to original datafile
with h5py.File(dataH5_path,'a') as h5file:
    if f'catalog/cat_by_sta/{config["station"]}' in h5file.keys():
        del h5file[f"catalog/cat_by_sta/{config['station']}"]
    catalog_sta_group = h5file.create_group(f"catalog/cat_by_sta/{config['station']}")
    for col in cat_keep_sgram.columns:
        if col == 'datetime':
            catalog_sta_group.create_dataset(name='datetime',data=np.array(cat_keep_sgram['datetime'],dtype='S'))
        else:
            exec(f"catalog_sta_group.create_dataset(name='{col}',data=cat_keep_sgram.{col})")

# save local catalog to new ML datafile
with h5py.File(SpecUFEx_H5_path,'a') as h5file:
    if f'catalog/cat_by_sta/{config["station"]}' in h5file.keys():
        del h5file[f"catalog/cat_by_sta/{config['station']}"]
    catalog_sta_group = h5file.create_group(f"catalog/cat_by_sta/{config['station']}")
    for col in cat_keep_sgram.columns:
        if col == 'datetime':
            catalog_sta_group.create_dataset(name='datetime',data=np.array(cat_keep_sgram['datetime'],dtype='S'))
        else:
            exec(f"catalog_sta_group.create_dataset(name='{col}',data=cat_keep_sgram.{col})")