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
    total_number_wells = d_images.shape[0]
    
    support = supp_methods.create_2d_support(shapes, exclude, labels.shape)
    total_number_chunks = support.shape[0]*support.shape[1]
    support_1col = support.reshape(1,-1).T
    labels_1col = labels.reshape(1,-1).T
    matlab_1col = matlab.reshape(1,-1).T
    included_ind_1_col=np.nonzero(support_1col)
    labels_1col_included = labels_1col[included_ind_1_col[0],:]
    matlab_1col_included = matlab_1col[included_ind_1_col[0],:]
    included_number_wells = np.sum(np.logical_not(exclude))
    included_number_chunks= int(np.sum(support.flatten()))
    metric1=supp_methods.calc_eval_metric(matlab_1col_included, labels_1col_included, True, dataset_lb_shape)
    metric2 = supp_methods.calc_add_metric(matlab, labels, exclude)
    #(a,b,c) = supp_methods.random_divide_sample_chunks(included_number_chunks, 0.6,0.2,0.2)
    (d,e,f,_,_,_) = supp_methods.random_divide_samples(support, exclude, 0.6,0.2)
    
