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


def exclude_bool(to_load):
    import h5py
    with h5py.File(to_load, "a", libver="latest") as f1:
         feat2 = f1["Raw_data/exclude_dataset"]
        
         feat2_array = feat2[:]
         del f1["Raw_data/exclude_dataset"]
         #import pdb; pdb.set_trace()
         feat2_bool = np.bool_(feat2_array)
         f1.create_dataset("Raw_data/exclude_dataset", compression=7, data=feat2_bool)
         # print(feat2_new.shape)
         # print(feat1.shape)

exclude_bool(datadir+'0_raw_dataset_384bact0freez31_aug1.hdf5')
exclude_bool(datadir+'0_raw_dataset_384bact0freez31_aug2.hdf5')
exclude_bool(datadir+'1_raw_dataset_384water1freez31f2_aug1.hdf5')

