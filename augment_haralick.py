#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 20:34:41 2019

@author: andrey
"""
import h5py
import numpy as np
import imutils
import cv2

hd5py_dir = "/data/Freezing_samples/h5data_new/"
# just for example
dataset_to_read = hd5py_dir+'0_raw_dataset_384bact0freez31.hdf5'
dataset_to_write = hd5py_dir+'0_raw_dataset_384bact0freez31_aug.hdf5'
#def augment_data(dataset_to_read, dataset_to_write):
with h5py.File(dataset_to_read, "r", libver="latest") as f1, h5py.File(dataset_to_write, "w", libver="latest") as f2:
    # so we will read each imge from the dataset one by one and then 
    g2 = f2.create_group('Raw_data')
    images_d = f1['Raw_data/images_dataset']
    images_shape = images_d.shape
    
    None
