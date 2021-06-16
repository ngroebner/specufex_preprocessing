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
import yaml

from matplotlib import pyplot as plt

# from generators import gen_sgram_QC

import tables
tables.file._open_files.close_all()

from specufex import BayesianNonparametricNMF, BayesianHMM


# load project variables: names and paths
key = sys.argv[1]

config_filename = sys.argv[1]
with open(config_filename, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

dataH5_name = f'data_{config["name"]}.hdf5'
dataH5_path = config['pathProj'] + 'H5files/' + dataH5_name
SpecUFEx_H5_name = f'SpecUFEx_{config["name"]}.hdf5'
SpecUFEx_H5_path = config['pathProj'] + '/H5files/' + SpecUFEx_H5_name
sgramMatOut = config['pathProj'] + 'matSgrams/'## for testing
pathWf_cat  = config['pathProj'] + 'wf_cat_out.csv'
pathSgram_cat = config['pathProj'] + f'sgram_cat_out_{config["name"]}.csv'

sgram_cat = pd.read_csv(pathSgram_cat)
# get spectrograms from H5
X= []
H5=True

if H5:
    with h5py.File(SpecUFEx_H5_path,'a') as fileLoad:
        for evID in fileLoad['spectrograms']:
            specMat = fileLoad['spectrograms'].get(evID)[:]
            Xis = specMat
            X.append(Xis)

X = np.array(X)

# NMF
print("Fitting NMF")
nmf = BayesianNonparametricNMF(X.shape)
nmf.fit(X, verbose=1)
print("NMF fit.")
print("Creating activation matrices.")
Vs = nmf.transform(X)
print("Activation matrices created.")

# HMM
print("Fitting HMM")
hmm = BayesianHMM(nmf.num_pat, nmf.gain)
hmm.fit(Vs)
print("HMM fit")
print("Transforming activation matrices and creating fingerprints.")
fingerprints, As, gams = hmm.transform(Vs)
print("Fingerprints created.")

# Plotting
plt.imshow(fingerprints[0])

# =============================================================================
# save output to H5
# =============================================================================
print("Saving results and models to HDF5 file")
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
        fp_group.create_dataset(name= str(evID), data=fingerprints[i])
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

print('Done.')