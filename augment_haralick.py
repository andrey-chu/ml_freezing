#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 20:34:41 2019

@author: andrey
"""
import h5py
import numpy as np

hd5py_dir = "/data/Freezing_samples/h5data_new/"
# just for example
dataset_to_read = hd5py_dir+'0_raw_dataset_384bact0freez31.hdf5'
dataset_to_write = hd5py_dir+'0_raw_dataset_384bact0freez31_aug.hdf5'
with h5py.File(dataset_to_read, "r", libver="latest") as f1, h5py.File(dataset_to_write, "w", libver="latest") as f2:
    
    None
