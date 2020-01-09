#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:01:53 2020

@author: andrey
"""

import platform
import numpy as np
if platform.node()=='choo-desktop':
    from branch_init_choo import datadir
elif platform.node()=='andrey-cfin':
    from branch_init_cfin import datadir
import h5py
to_check = datadir+'../h5data_backup/0_raw_dataset_384bact0freez31f2.hdf5'
well = 0
featurenum =0
with h5py.File(to_check, "r", libver="latest") as f1:
    feat2 = f1["Raw_data/features2_dataset"]
    feat2_array = feat2[:]
    aa = feat2_array[well, featurenum, :]