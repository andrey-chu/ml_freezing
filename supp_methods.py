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
    #confusion_matrix_mine1 = confusion_matrix_mine(estimated_labels,gt_labels, labels)
    # classification report
    class_report_dict = skl_met.classification_report(estimated_labels, gt_labels,labels\
                                                      , output_dict=True)
    precision_calc = class_report_dict["weighted avg"]["precision"]
    recall_calc = class_report_dict["weighted avg"]["recall"]
    F1_calc = class_report_dict["weighted avg"]["f1-score"]
    if printinfo:
        print(confusion_matrix_skl)
     #   print(confusion_matrix_mine1)
    
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
    This function gives more flexible division into sets, but it could happen, 
    that it will divide in such a way so in training will be no freezing points,
    so not recommended.
    Note difference from the other function, it uses all the given chunks, so 
    the "exclusion" has to be performed before
    """
    import numpy as np
    #import random
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
def random_divide_samples(support, exclude, prop_training, prop_cv):
    import numpy as np
    #import random
    """
    The function randomly divides wells between train, cv and test
    returns tuple of 6 bool arrays, three for chunks, three for wells
    """
    #included_number_chunks= int(np.sum(support.flatten()))
    training_set_wells = np.zeros((1,exclude.shape[1]), dtype=np.bool_)
    cv_set_wells = np.zeros((1,exclude.shape[1]), dtype=np.bool_)
    test_set_wells = np.zeros((1,exclude.shape[1]), dtype=np.bool_)
    included_number_wells = np.sum(np.logical_not(exclude))
    wells =  np.where(np.logical_not(exclude))[1]
    np.random.shuffle(wells)
    well_support = np.zeros(support.shape)
    training_num = int(prop_training*included_number_wells)
    cv_num = int(prop_cv*included_number_wells)
    training_wells = wells[:training_num]
    cv_wells = wells[training_num:training_num+cv_num]
    test_wells = wells[training_num+cv_num:]
    well_support_training=well_support.copy()
    well_support_training[:,training_wells]=1
    well_support_cv=well_support.copy()
    well_support_cv[:,cv_wells]=1
    well_support_test=well_support.copy()
    well_support_test[:,test_wells]=1
    well_support_training=(well_support_training*support).reshape(1,-1).T
    well_support_cv=(well_support_cv*support).reshape(1,-1).T
    well_support_test=(well_support_test*support).reshape(1,-1).T
    training_set_wells[0,training_wells]=1
    cv_set_wells[0,cv_wells]=1
    test_set_wells[0,test_wells]=1
    return (well_support_training, well_support_cv, well_support_test,\
            training_set_wells,cv_set_wells,test_set_wells)
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
    maxdim=len(shapes.shape)-1
    for i in range(shape_2d[1]):
        shapes_to_support[int(shapes[i,maxdim]):, i] = 0
    support = support*shapes_to_support
    return support

def create_2d_support_chunked(shapes, excludes, shape_2d):
    import numpy as np
    # first let's check that the dimensions are right
    assert(shape_2d[1]==shapes.shape[0])&(excludes.shape[1]==shape_2d[1])
    support = np.ones(shape_2d)
    support[:,np.nonzero(excludes)[1]] = False
    shapes_to_support = np.ones(shape_2d)
    maxdim=shapes.shape[1]-1
    for i in range(shape_2d[1]):
        shapes_to_support[int(shapes[i,maxdim]):, i] = 0
    support = support*shapes_to_support
    return support


def transform_back(total_shape, positions_2d, labels):
    import numpy as np
    new_support = np.empty(total_shape)
    new_support.fill(np.nan)
    new_support[positions_2d] = labels
    return new_support
    
def find_freezing_by_frozen(predicted_array, wells_set):
    import numpy as np
    if (wells_set.ndim == 1):
        wells_set_list = wells_set
    elif (wells_set.ndim == 2):
        wells_set_list = np.nonzero(wells_set)[1]
    freezing_points = np.zeros((1,len(wells_set_list)))
    length_array = predicted_array.shape[0]
    ii = 0
    for i in wells_set_list:
        
        array_i=predicted_array[:,i]
        reversed_array = array_i[::-1]
        if np.sum(array_i==1)==0:
        # if there is no 1's in array
            itera = 0
            j = reversed_array[itera]
            if np.sum(array_i==3)>0:
                while (j!=3 and j!=2 and not(np.isnan(j))):
                    itera +=1
                    j=reversed_array[itera]
            # let's go from the end
            if np.sum(reversed_array==2)==0:
                # if there are no 2's in aray 
                # we go back until we encounter anything which is not 3 or nan
                while (j==3 or (np.isnan(j))):
                    itera +=1
                    j=reversed_array[itera]
            else:
                # if there are 2's in array we go back until the first 2
                while (j!=2):
                    itera +=1
                    j=reversed_array[itera]
            #reversed_array[itera] =1
            predicted_array[-itera,i] =1
            freezing_points[0,ii] = length_array-itera
        else:
            # if there are 1's than the first one is the freezing point
            freezing_points[0,ii] = np.min(np.nonzero(array_i==1)[0])
        ii+=1
    return (freezing_points, predicted_array)

def seg_find_freezing_by_frozen(predicted_array1):
    import numpy as np
    predicted_array = predicted_array1.copy()
    freezing_points = np.zeros((1,len(predicted_array)))
    #import pdb; pdb.set_trace()
    ii = 0
    for i in range(len(predicted_array)):
        length_array = predicted_array[i].shape[0]
        array_i=predicted_array[i]
        reversed_array = array_i[::-1]
        if np.sum(array_i==1)==0:
        # if there is no 1's in array
            itera = 0
            j = reversed_array[itera]
            if np.sum(array_i==3)>0:
                while (j!=3 and j!=2 and not(np.isnan(j))):# added j!= for those cases when 2's come after 3's
                    itera +=1
                    j=reversed_array[itera]
            # let's go from the end
            if np.sum(reversed_array==2)==0:
                # if there are no 2's in aray 
                # we go back until we encounter anything which is not 3 or nan
                while (j==3 or (np.isnan(j))):
                    itera +=1
                    j=reversed_array[itera]
            else:
                # if there are 2's in array we go back until the first 2
                while (j!=2):
                    itera +=1
                    j=reversed_array[itera]
            #reversed_array[itera] =1
            predicted_array[i][-itera] =1
            freezing_points[0,ii] = length_array-itera
        else:
            # if there are 1's than the first one is the freezing point
            freezing_points[0,ii] = np.min(np.nonzero(array_i==1)[0])
        ii+=1
    return (freezing_points, predicted_array)
        
def freezing_est_statistic(labels_mat, wells):
    # wells in columns
    import numpy as np
    labels_short_mat = labels_mat[:,np.nonzero(wells)[1]]
    # What if there are two ones in one column, we have to find a way to find a first(?) one
    return np.argmax(labels_short_mat==1, axis=0)
def freezing_metrics(freezepoints, freezepoints_gt, thresh):
    #thresh is a threshold where freezing points regarded to be the same
    import numpy as np
    difference = (freezepoints - freezepoints_gt)
    errs = np.abs(difference)<thresh
    err = np.mean(errs)
    mean_dist = np.mean(np.abs(difference))
    return (err, mean_dist, errs)

def extract_haralick(images_d):
    
    # the method gets the image database, reads it and outputs the features
    import mahotas as mt
    import numpy as np

    
    print("Extracting Haralick features")
    # features1=np.mean(mt.features.haralick(image), axis=0)
    image_shape= images_d.shape
    # import pdb; pdb.set_trace()
    features = np.empty((image_shape[3], 13, image_shape[0])) # the 14th 
                # feature is not given in the lib, we should calculate it ourselves
                # if needed
    for i in range(image_shape[0]):
        print("Extracting progress: {0:6.2f}%".format(100*i/image_shape[0]))
        for j in range(image_shape[3]):
            #print(str(j)+" out of "+str(image_shape[3]))
            image = images_d[i,:,:,j]
            features[j,:,i]=np.mean(mt.features.haralick(image), axis=0)
    return features

def extract_haralick_parallel(images_d, cores_num):
    
    # the method gets the image database, reads it and outputs the features
    import mahotas as mt
    import numpy as np
    from joblib import Parallel, delayed
    def haralik_feat(image_stk):
        features1=np.empty((image_stk.shape[2], 13))
        for j in range(image_stk.shape[2]):
            features1[j,:]=np.mean(mt.features.haralick(image_stk), axis=0)
        return features1
    print("Extracting Haralick features")
    # features1=np.mean(mt.features.haralick(image), axis=0)
    image_shape= images_d.shape
    # import pdb; pdb.set_trace()
    features_array = np.empty((image_shape[3], 13, image_shape[0])) # the 14th 
                # feature is not given in the lib, we should calculate it ourselves
                # if needed
    features=Parallel(n_jobs=cores_num, verbose=5)(delayed(haralik_feat)(images_d[i,:,:,:])  for i in range(image_shape[0]))
    #import pdb; pdb.set_trace()
    for i in len(features):
        features_array[i, :, :] =features[i].T
    return features_array