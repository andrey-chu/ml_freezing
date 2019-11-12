#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 00:30:17 2019
Here I will stock some functions that can be called from the various files
@author: andrey
"""

def calc_eval_metric(estimated_labels, gt_labels, printinfo, original_shape):
    """
    The function returns the quality of the labels versus the ground truth
    it returns a tuple of (precision, recall, F1)
    and also prints confusion matrices out if printinfo is given true
    """
    import numpy as np
    import sklearn.metrics as skl_met
    labels=np.union1d(np.unique(gt_labels),np.unique(estimated_labels))
    # confusion matrix
    confusion_matrix_skl=skl_met.confusion_matrix(estimated_labels, gt_labels)
    confusion_matrix_mine1 = confusion_matrix_mine(estimated_labels,gt_labels, labels)
    # classification report
    class_report_dict = skl_met.classification_report(estimated_labels, gt_labels,labels\
                                                      , output_dict=True)
    precision_calc = class_report_dict["weighted avg"]["precision"]
    recall_calc = class_report_dict["weighted avg"]["recall"]
    F1_calc = class_report_dict["weighted avg"]["f1-score"]
    if printinfo:
        print(confusion_matrix_skl)
        print(confusion_matrix_mine1)
    
    return (precision_calc, recall_calc, F1_calc)
def calc_add_metric(estimated_mat, gt_mat, exclude):
    import numpy as np
    import pandas as pd
    """
    Calculate additional metric
    This function calculates how far is freezing point from the gt value
    It also calculates number of "unphysical failures"(assuming temperatures 
    are decreasing monotonically):
        - type 1 if there is more than one freezing point
        - type 2 if there is water in lower temperature than ice
        - type 3 if there is freezing point at lower temperature than ice
        - alltypes counts as one if any of previous is one
    unphysical marking: the array of the size of wells where 1 denotes a 
    problematic well for this estimation
    """
    estimated_mat_fp_only = estimated_mat.copy()
    gt_mat_fp_only = gt_mat.copy()
    
    estimated_mat_fp_only[estimated_mat_fp_only>1]=0
    gt_mat_fp_only[gt_mat_fp_only>1]=0
    estimated_water_points = np.zeros(estimated_mat.shape)
    estimated_water_points[np.where(estimated_mat==0)]=1
    estimated_water_points_df = pd.DataFrame(estimated_water_points)
    estimated_ice_points = np.zeros(estimated_mat.shape)
    estimated_ice_points[np.where(estimated_mat==3)]=1
    estimated_ice_points_df = pd.DataFrame(estimated_ice_points)
    gt_fp = pd.DataFrame(gt_mat_fp_only)
    estimated_fp = pd.DataFrame(estimated_mat_fp_only)
    #gt_fp_number_per_row = gt_fp.astype(bool).sum(axis=0)
    estimated_fp_number_per_row = estimated_fp.astype(bool).sum(axis=0)
    gt_freezing_points = np.array(gt_fp[gt_fp>0].idxmax())
    estimated_freezing_points = np.array(estimated_fp[estimated_fp>0].idxmax())
    estimated_maxwater_points = estimated_water_points_df.apply(lambda series: series[series>0].last_valid_index())
    estimated_minice_points = estimated_ice_points_df.apply(lambda series: series[series>0].first_valid_index())
    ### Now freezing point distances
    fp_distance_mean = np.mean(np.abs((estimated_freezing_points[np.where(np.logical_not(exclude))[1]]-gt_freezing_points[np.where(np.logical_not(exclude))[1]])))
    fp_distance_median = np.median(np.abs((estimated_freezing_points[np.where(np.logical_not(exclude))[1]]-gt_freezing_points[np.where(np.logical_not(exclude))[1]])))
    type2err = np.sum((estimated_minice_points[np.where(np.logical_not(exclude))[1]]-estimated_maxwater_points[np.where(np.logical_not(exclude))[1]])<0)
    type3err = np.sum((estimated_minice_points[np.where(np.logical_not(exclude))[1]]-estimated_freezing_points[np.where(np.logical_not(exclude))[1]])<0)
    type1err = np.sum(estimated_fp_number_per_row[np.where(np.logical_not(exclude))[1]] !=1)
    return (fp_distance_mean,fp_distance_median,type1err,type2err,type3err)
    #freezing_distances = 
def random_divide_sample_chunks(number_of_chunks, prop_training, prop_cv, prop_test):
    """
    The function randomly divides chunks between train, cv and test
    returns tuple of three bool arrays
    """
    import numpy as np
    import random
    training_set = np.zeros((1,number_of_chunks), dtype=np.bool_)
    cv_set = np.zeros((1,number_of_chunks), dtype=np.bool_)
    test_set = np.zeros((1,number_of_chunks), dtype=np.bool_)
    chunks = np.arange(number_of_chunks)
    np.random.shuffle(chunks)
    training_num = int(prop_training*number_of_chunks)
    cv_num = int(prop_cv*number_of_chunks)
    #test_num = number_of_chunks - training_num - cv_num
    training_chunks = chunks[:training_num]
    cv_chunks = chunks[training_num:training_num+cv_num]
    test_chunks = chunks[training_num+cv_num:]
    training_set[:,training_chunks]=True
    cv_set[:,cv_chunks]=True
    test_set[:,test_chunks]=True
    return (training_set, cv_set, test_set)
def random_divide_samples(support, exclude, percent_training, percent_cv, percent_test):
    import numpy as np
    import random
    """
    The function randomly divides wells between train, cv and test
    returns tuple of 6 bool arrays, three for chunks, three for wells
    """
    included_number_chunks= int(np.sum(support.flatten()))
    training_set = np.zeros((1,included_number_chunks), dtype=np.bool_)
    cv_set = np.zeros((1,included_number_chunks), dtype=np.bool_)
    test_set = np.zeros((1,included_number_chunks), dtype=np.bool_)
    included_number_wells = np.sum(np.logical_not(exclude))
    wells =  np.where(np.logical_not(exclude))[1]
    np.random.shuffle(wells)
    well_support = np.zeros(support.shape)
    training_num = int(prop_training*included_number_wells)
    cv_num = int(prop_cv*included_number_wells)
    training_wells = wells[:training_num]
    cv_wells = wells[training_num:training_num+cv_num]
    test_wells = wells[training_num+cv_num:]
    
def confusion_matrix_mine(estimated_labels, gt_labels, labels):
    import numpy as np
    matrix1 = np.zeros((len(labels),len(labels)))
    for i in range(len(labels)):
        for j in range(len(labels)):
            matrix1[i,j] = np.sum(np.logical_and(estimated_labels==labels[i],gt_labels==labels[j]))
    return matrix1

def create_2d_support(shapes, excludes, shape_2d):
    import numpy as np
    # first let's check that the dimensions are right
    assert(shape_2d[1]==shapes.shape[0])&(excludes.shape[1]==shape_2d[1])
    support = np.ones(shape_2d)
    support[:,np.nonzero(excludes)[1]] = False
    shapes_to_support = np.ones(shape_2d)
    for i in range(shape_2d[1]):
        shapes_to_support[int(shapes[i,2]):, i] = 0
    support = support*shapes_to_support
    return support
    