#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 23:51:35 2019
This file implements the SVM on the freezing essay (the chunked database)
In particular in this case we will only use haralick features as they were
acquired previously in matlab and stored in the database
@author: andrey
"""
import h5py
import numpy as np
import supp_methods

datadir = "/data/Freezing_samples/h5data/"
chunked_united_dataset = datadir + "united_chunked_dataset_96.hdf5"

with h5py.File(chunked_united_dataset, 'r') as f:
    d_images = f["Raw_data/images_dataset"]
    d_labels = f["Raw_data/labels_dataset"]
    d_features = f["Raw_data/features_dataset"]
    d_matlab = f["Raw_data/matlab_dataset"]
    d_exclude=f["Raw_data/exclude_dataset"]
    d_shapes=f["Raw_data/shapes_dataset"]
    shapes = d_shapes[:]
    exclude = d_exclude[:]
    dataset_im_shape = d_images.shape
    dataset_lb_shape = d_labels.shape
    labels = d_labels[:]
    matlab = d_matlab[:]
    support = supp_methods.create_2d_support(shapes, exclude, labels.shape)
    support_1col = support.flatten()
    d_labels_1col = labels.flatten()
    d_matlab_1col = matlab.flatten()
    d_labels_1col_included = d_labels_1col(np.nonzero(support_1col)[1])
    metric1=supp_methods.calc_eval_metric(d_matlab_1col, d_labels_1col, True, dataset_lb_shape)
