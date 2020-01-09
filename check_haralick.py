#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:15:04 2020

@author: andrey
"""

import platform
import h5py
#import mahotas as mt

from matplotlib import pyplot as plt
if platform.node()=='choo-desktop':
    from branch_init_choo import datadir
elif platform.node()=='andrey-cfin':
    from branch_init_cfin import datadir

dataset_to_read1 = datadir+'0_raw_dataset_384bact0freez31f2.hdf5'
dataset_to_read2 = datadir+'0_raw_dataset_384bact0freez31_aug.hdf5'
dataset_to_read3 = datadir+'united_raw_dataset_384freez31f2.hdf5'
with h5py.File(dataset_to_read1, "r", libver="latest") as f1, h5py.File(dataset_to_read2, "r", libver="latest") as f2:
    features_d1 = f1['Raw_data/features2_dataset']
    features_d11 = f1['Raw_data/features_dataset']
    features_d2 = f2['Raw_data/features2_dataset']
    print(features_d1.shape)
    length_feat = features_d1.shape[0]
    wells_num = features_d1.shape[2]
    feature_num = 12
    
    angle = 0
    well = 183
    before_rotation = features_d1[:,feature_num,well]
    after_rotation = features_d2[:,feature_num,well+angle*wells_num]
    print(features_d11.shape)
    print(features_d2.shape)
    plt.figure()
    plt.title('Haralick comparison')
    plt.grid()
    plt.plot(range(length_feat), before_rotation, '-', color="g",
             label="Feature {0} before rotation".format(feature_num))
    plt.plot(range(length_feat), after_rotation, '-', color="r",
             label="Feature {0} after rotation".format(feature_num))
    plt.legend(loc="best")
    
    
    