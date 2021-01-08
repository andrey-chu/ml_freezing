#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:14:00 2020

@author: andrey
"""
import platform
import numpy as np
if platform.node()=='choo-desktop':
    from branch_init_choo import datadir
elif platform.node()=='andrey-cfin':
    from branch_init_cfin import datadir
elif platform.node()=='andrey-workbook':
    from branch_init_laptop import datadir

def swapaxes_feature2(to_load):
    import h5py
    with h5py.File(to_load, "a", libver="latest") as f1:
         feat2 = f1["Raw_data/features2_dataset"]
         feat1 = f1["Raw_data/features_dataset"]
         feat2_array = feat2[:]
         del f1["Raw_data/features2_dataset"]
         feat2_array_reshaped = np.swapaxes(feat2_array, 0, 2)
         feat2_new=f1.create_dataset("Raw_data/features2_dataset", compression=7, data=feat2_array_reshaped)
         print(feat2_new.shape)
         print(feat1.shape)

#swapaxes_feature2(datadir+'0_raw_dataset_384bact0freez31f2.hdf5')
swapaxes_feature2(datadir+'1_raw_dataset_384water1freez31f2.hdf5')
# swapaxes_feature2(datadir+'0_raw_dataset_96bact0freez31f2.hdf5')
# swapaxes_feature2(datadir+'1_raw_dataset_96bact1freez31f2.hdf5')
# swapaxes_feature2(datadir+'2_raw_dataset_96water2freez31f2.hdf5')