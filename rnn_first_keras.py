#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:51:57 2021

@author: andrey
"""
import platform
import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import LSTM
"""
Let us load the data first
We  need the raw features and the output
"""
if platform.node()=='choo-desktop':
    from branch_init_choo import datadir
elif platform.node()=='andrey-cfin':
    from branch_init_cfin import datadir
elif platform.node()=='andrey-workbook':
    from branch_init_laptop import datadir


model = Sequential()
model.add(LSTM(1, input_shape=(timesteps, data_dim), return_sequences=True))

raw_united_dataset = datadir + 'united_raw_dataset_384freez31f2_w_aug.hdf5'
with h5py.File(raw_united_dataset, 'r') as f:
    d_images = f["Raw_data/images_dataset"]
    d_labels = f["Raw_data/labels_dataset"]
    d_features = f["Raw_data/features_dataset"]
    d_features2 = f["Raw_data/features2_dataset"]
    d_matlab = f["Raw_data/matlab_dataset"]
    d_exclude=f["Raw_data/exclude_dataset"]
    d_shapes=f["Raw_data/shapes_dataset"]

