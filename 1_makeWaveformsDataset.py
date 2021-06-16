#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:00:19 2021
example: Parkfield repeaters::
@author: theresasawi
"""



import h5py
import numpy as np
import glob
import sys
import obspy
import os
import pandas as pd
import yaml

sys.path.append('functions/')
from generators import gen_wf_from_folder


#%% load project variables: names and paths


config_filename = sys.argv[1]
with open(config_filename, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

#pathProj, pathCat, pathWF, network, station, channel, channel_ID, filetype, cat_columns = setParams(key)
pathProj = config['pathProj']
pathCat = pathProj + config['pathCat']
pathWF = pathProj + config['pathWF']

# should probably use the path building functions in case
# a user does or doens't add a trailing slash to the
# paths in the config file
dataH5_name = f'data_{config["name"]}.hdf5'

dataH5_path = pathProj + '/H5files/' + dataH5_name

wf_cat_out = pathProj + 'wf_cat_out.csv'


if not os.path.isdir(pathProj + '/H5files/'):
    os.mkdir(pathProj + '/H5files/')

#%% get global catalog

cat = pd.read_csv(pathCat, header=None,delim_whitespace=True)


cat.columns = config['cat_columns']

#for plotting in later scripts
try:
    cat['datetime'] = pd.to_datetime(cat[['year','month','day','hour','minute','second']])
except:
    print('YOU SHOULD MAKE A DATETIME COLUMN FOR ANALYSIS LATER!')
    pass

cat['event_ID'] = [int(evID) for evID in cat.event_ID]

print('event ID: ', cat.event_ID.iloc[0])



#%% get list of waveforms and sort

wf_filelist = glob.glob(pathWF + '*')
wf_filelist.sort()

wf_filelist = wf_filelist

wf_test = obspy.read(wf_filelist[0])

lenData = len(wf_test[0].data)

#%% define generator (function)


gen_wf = gen_wf_from_folder(wf_filelist,key,lenData,channel_ID)


## clear old H5 if it exists, or else error will appear
if os.path.exists(dataH5_path):
    os.remove(dataH5_path)

#%% add catalog and waveforms to H5


evID_keep = [] #list of wfs to keep

with h5py.File(dataH5_path,'a') as h5file:



    global_catalog_group =  h5file.create_group("catalog/global_catalog")


    for col in cat.columns:

        if col == 'datetime': ## if there are other columns in your catalog
        #that are stings, then you may need to extend conditional statement
        # to use the dtype='S' flag in the next line
            global_catalog_group.create_dataset(name='datetime',data=np.array(cat['datetime'],dtype='S'))

        else:
            exec(f"global_catalog_group.create_dataset(name='{col}',data=cat.{col})")


    waveforms_group  = h5file.create_group("waveforms")
    station_group = h5file.create_group(f"waveforms/{station}")
    channel_group = h5file.create_group(f"waveforms/{station}/{channel}")



    dupl_evID = 0 #duplicate event IDs?? not here, sister
    n=0

    while n <= len(wf_filelist): ## not sure a better way to execute this? But it works

        try:   #catch generator "stop iteration" error


            #these all defined in generator at top of script
            data, evID, n = next(gen_wf)
            if n%500==0:
                print(n, '/', len(wf_filelist))
            # if evID not in group, add dataset to wf group
            if evID not in channel_group:
                channel_group.create_dataset(name= evID, data=data)
                evID_keep.append(int(evID))
            elif evID in channel_group:
                dupl_evID += 1

        except StopIteration: #handle generator error
            break


    sampling_rate = wf_test[0].stats.sampling_rate
    # instr_response = wf_test[0].stats.instrument_response
    station_info = f"{wf_test[0].stats.network}.{wf_test[0].stats.station}.{wf_test[0].stats.location}.{wf_test[0].stats.channel}."
    calib = wf_test[0].stats.calib
    _format = wf_test[0].stats._format


    processing_group = h5file.create_group(f"{station}/processing_info")


    processing_group.create_dataset(name= "sampling_rate_Hz", data=sampling_rate)#,dtype='S')
    processing_group.create_dataset(name= "station_info", data=station_info)
    processing_group.create_dataset(name= "calibration", data=calib)#,dtype='S')
    processing_group.create_dataset(name= "orig_formata", data=_format)#,dtype='S')
    # processing_group.create_dataset(name= "instr_response", data=instr_response,dtype='S')
    processing_group.create_dataset(name= "lenData", data=lenData)#,dtype='S')






print(dupl_evID, ' duplicate events found and avoided')
print(n- dupl_evID, ' waveforms loaded')



#%% save final working catalog to csv


cat_keep_wf = cat[cat['event_ID'].isin(evID_keep)]

if os.path.exists(wf_cat_out):
    os.remove(wf_cat_out)


cat_keep_wf.to_csv(wf_cat_out)

print(len(cat_keep_wf), ' events in wf catalog')


#%%
