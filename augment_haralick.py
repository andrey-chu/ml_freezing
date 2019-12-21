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
ang_num =20
angles = np.arange(0,360,360/ang_num)
with h5py.File(dataset_to_read, "r", libver="latest") as f1, h5py.File(dataset_to_write, "w", libver="latest") as f2:
    images_d = f1['Raw_data/images_dataset']
#    position_d = f1['Raw_data/positions_dataset']
#    temperatures_d =f1['Raw_data/temperatures_dataset']
    labels_d = f1['Raw_data/labels_dataset']
#    features_d = f1['Raw_data/features_dataset']
#    matlab_d = f1['Raw_data/matlab_dataset']
#    substance_d = f1['Raw_data/substance_dataset']
#    exclude_d = f1['Raw_data/exclude_dataset']
#    "Raw_data/datasets_dataset"
    im_shape = images_d.shape
    for i in range(im_shape[3]):
        imstack = images_d[:,:,:,i]
        image = imstack[:,:,1]
        for angle in angles:
            rotated = imutils.rotate(image, angle)
            cv2.imshow("Rotated (Problematic)", rotated)
            cv2.waitKey(0)
    
