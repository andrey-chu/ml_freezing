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
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from seglearn.pipe import Pype
from seglearn.transform import FeatureRep, SegmentX, SegmentXY 
from sklearn.metrics import f1_score, make_scorer
from seglearn.base import TS_Data
from seglearn.transform import FunctionTransformer
from seglearn.feature_functions import mean, median, abs_energy, std, skew, mean_crossings, minimum, maximum, mean_diff,\
zero_crossing, var
from supp_methods import seg_find_freezing_by_frozen, extract_haralick
import platform
if platform.node()=='choo-desktop':
    from branch_init_choo import datadir
elif platform.node()=='andrey-cfin':
    from branch_init_cfin import datadir

def features2seg(features, shapes):
    return [features[:int(shapes[i,1]),:,i] for i in range(features.shape[2])]
def reshape_all(Xt):
    return Xt.reshape((Xt.shape[0],-1))
    
def labels2seg(labels, shapes):
    return [labels[:int(shapes[i,1]),i] for i in range(labels.shape[1])]

def change_labels(labels, number, freezing_length=20):
    # input: labels in timepoints x wells
    #       number: if 2 then labels changed into below freezing (0) above freezing (1)
    #               if 3 then below freezing (0), freezing (2), above freezing (1)
    # assumes: in input the first 1 is at the freezing point
    changed_labels = np.ones(labels.shape)
    fp = np.argmax(labels==1, axis=0)+1
    if number == 2:
        for i in range(labels.shape[1]):
            changed_labels[:fp[i],i] =0
    elif number==3:
        for i in range(labels.shape[1]):
            changed_labels[:fp[i]-freezing_length,i] =0
            changed_labels[fp[i]-freezing_length:fp[i],i] =2
    return changed_labels

def change_labels_preserve(labels, number, freezing_length=31):
    # input: labels in timepoints x wells
    #       number: if 2 then labels changed into below freezing (0) above freezing (3)
    #               if 3 then below freezing (0), freezing (2), above freezing (3)
    # assumes: in input the first 1 is at the freezing point
    changed_labels = np.ones(labels.shape)*3
    fp = np.argmax(labels==1, axis=0)+1
    if number == 2:
        for i in range(labels.shape[1]):
            changed_labels[:fp[i],i] =0
    elif number==3:
        for i in range(labels.shape[1]):
            changed_labels[:fp[i]-freezing_length,i] =0
            changed_labels[fp[i]-freezing_length:fp[i],i] =2
    return changed_labels


def fill_in_nans(npstring):
    for i in range(npstring.shape[0]-1,-1,-1):
        if np.isnan(npstring[i]):
            npstring[i] = npstring[i+1]
    return npstring

def inverse_transform(seg, unsegmented, width, step1, order='F'):
    #transforms back from the segmented output to the seg-input
    #basically an inverse transform of SegmentXY
    #‘C’ means C-like index order (first index changes slowest) 
    #‘F’ means Fortran-like index order (last index changes slowest). 
    inv_transformed = [None]*len(unsegmented)
    itera = 0
    for i in range(len(unsegmented)): # going by the list
        new_one=np.empty(unsegmented[i].shape)
        new_one[:] = np.nan
        new_el_number=(unsegmented[i].shape[0]-width+1)//step1
        new_one[width-1::step1]= seg[itera:new_el_number+itera]
        new_one=fill_in_nans(new_one)
        inv_transformed[i]=new_one
        itera+=new_el_number
    return inv_transformed
def linear_f(size):
    return np.linspace(0,size,size)
fun_dict ={
        "lin":linear_f
        }
def add_features(prev_features, to_add):
    new_features = [None]*len(prev_features)
    for i in range(len(prev_features)):
        length_add = prev_features[i].shape[0]
        it = prev_features[i].shape[1]
        add_feat_length = len(to_add)
        # create an empty array
        new_features[i]=np.empty((prev_features[i].shape[0],prev_features[i].shape[1]+add_feat_length))
        # add existing features
        new_features[i][:,:prev_features[i].shape[1]] = prev_features[i]
        for funs in to_add:
            new_features[i][:,it]=linear_f(length_add)
            it+=1
    return new_features

    



datadir = "/data/Freezing_samples/h5data_new/"
united_dataset = datadir + "united_raw_dataset_96freez31.hdf5"
change_labels_bool =1
conservative = 0
with h5py.File(united_dataset, 'r') as f:
    d_images = f["Raw_data/images_dataset"]
    d_labels = f["Raw_data/labels_dataset"]
    d_features = f["Raw_data/features_dataset"]
    d_matlab = f["Raw_data/matlab_dataset"]
    d_exclude=f["Raw_data/exclude_dataset"]
    d_shapes=f["Raw_data/shapes_dataset"]
    d_datasets=f["Raw_data/datasets_dataset"]
    d_substance=f["Raw_data/substance_dataset"]
    
    #features_data = d_features[:]
    features_data = extract_haralick(d_images)
    shapes = d_shapes[:]
    exclude = d_exclude[:]
    dataset_im_shape = d_images.shape
    dataset_lb_shape = d_labels.shape
    labels = d_labels[:]
    matlab = d_matlab[:]
    datasets=d_datasets[:]
    substance = d_substance[:].flatten()
    
    total_number_wells = d_images.shape[0]
    
    support = supp_methods.create_2d_support(shapes, exclude, labels.shape)
    # if we use images: flatten images, reshape and turn to segform
    #images=d_images[:]
    #images_flattened = images.reshape(images.shape[0],-1,images.shape[-1])
    #new_images_seg = features2seg(images_flattened, shapes)
    #new_images_seg_included = [new_images_seg[i] for i in np.nonzero(np.logical_not(exclude))[1]]
    
    #(a,b,c) = supp_methods.random_divide_sample_chunks(included_number_chunks, 0.6,0.2,0.2)
    #(training_data,cv_data,test_data,train_set_wells,_,test_set_wells) = supp_methods.random_divide_samples(support, exclude, 0.6,0.2)
    # the format of time series to work with seglearn is 
    # first we will change labels into only 2 (3?)
    if change_labels_bool == 1:
        new_labels = change_labels_preserve(labels,3)
        new_matlab = change_labels_preserve(matlab,3)
    else:
        new_labels_bool = labels
        new_matlab = matlab
    # now let us turn all the data into a format suitable
    #chosen_features = [3]
    #chosen_features=[0,1,2,3,5, 6,7,8,9,13]
    chosen_features=[0,1,2,3,5, 6,7,8,9,12]
    new_features_seg = features2seg(features_data[:,chosen_features,:],shapes)
    #new_features_seg = add_features(new_features_seg, ['lin'])
    new_labels_seg = labels2seg(new_labels,shapes)
    new_matlab_seg = labels2seg(new_matlab, shapes)
    # now let us exclude the bad wells
    new_features_seg_included = [new_features_seg[i] for i in np.nonzero(np.logical_not(exclude))[1]]
    new_labels_seg_included = [new_labels_seg[i] for i in np.nonzero(np.logical_not(exclude))[1]]
    new_matlab_seg_included = [new_matlab_seg[i] for i in np.nonzero(np.logical_not(exclude))[1]]
    datasets_included = datasets[np.nonzero(np.logical_not(exclude))[1]]
    substance_included = substance[np.nonzero(np.logical_not(exclude))[1]]
    # for investigation
    # create a feature representation pipeline
    chosenwidth=6
    fts = {'mean': mean, 'var': var, 'std': std, 'skew': skew, 'mnx': mean_crossings, 'minimum':minimum, 'maximum':maximum, \
           'mean_diff':mean_diff}
    scorer1 = make_scorer(f1_score, average='macro')
    C_chosen = 77.42#12.915#77.42#grid1.best_params_["C"]
    gamma_chosen = 0.05994#grid1.best_params_["gamma"]
    n_estimators = 20
    #clf = Pype([('segment', SegmentXY(width=chosenwidth,step=1)), # in this context what is the difference with SegmentX?
    #        ('features', FeatureRep(fts)),
    #        ('scaler', StandardScaler()),
    #        ('rf', RandomForestClassifier(n_estimators=20))], scorer=scorer1)
    clf = Pype([('segment', SegmentXY(width=chosenwidth,step=1)), # in this context what is the difference with SegmentX?
           ('features', FunctionTransformer(reshape_all)),
            ('scaler', StandardScaler()),
            ('bagg', OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', gamma=gamma_chosen, C=C_chosen, probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, warm_start=True, n_estimators=n_estimators, n_jobs=6, verbose=10)))])
            #scorer=scorer1
    X_train, X_test, y_train, y_test, matlab_train, matlab_test = train_test_split(new_features_seg_included, new_labels_seg_included, new_matlab_seg_included, test_size=0.10, random_state=10)

   
    clf.fit(X_train, y_train)
    #clf1.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    predict = clf.predict(X_test)
    test1, test2 = clf.transform_predict(X_test, y_test)
    train1, train2 = clf.transform_predict(X_train, y_train)
    test21= inverse_transform(test2, y_test, chosenwidth, 1, order='F')
    test11= inverse_transform(test1, y_test, chosenwidth, 1, order='F') 
    train11= inverse_transform(train1, y_train, chosenwidth, 1, order='F')
    train21= inverse_transform(train2, y_train, chosenwidth, 1, order='F')
    
    # and do a cross validation

    scoring = make_scorer(f1_score, average='macro')
    if change_labels ==1 and conservative ==1 :
        freeze2 = [np.argmax(test21[i]) for i in range(len(test21))]
        freeze2_conserve = [test21[i].shape[0]-np.argmax(np.flip(np.int_(np.logical_not(test21[i])))) for i in range(len(test21))]
        freeze_GT = [np.argmax(test11[i]) for i in range(len(test11))]# GT
        freeze_GT_conserve = [test11[i].shape[0]-np.argmax(np.flip(np.int_(np.logical_not(test11[i])))) for i in range(len(test11))]
        matlab_conserve = [matlab_test[i].shape[0]-np.argmax(np.flip(np.int_(np.logical_not(matlab_test[i])))) for i in range(len(matlab_test))]
        matlab_tr_conserve = [matlab_train[i].shape[0]-np.argmax(np.flip(np.int_(np.logical_not(matlab_train[i])))) for i in range(len(matlab_train))]
    else:

        freeze2, _ = seg_find_freezing_by_frozen(test21)
        freeze2_conserve = freeze2
        freeze_GT, _ = seg_find_freezing_by_frozen(test11)
        freeze_GT_conserve = freeze_GT
        matlab_conserve, _ = seg_find_freezing_by_frozen(matlab_test)
        matlab_tr_conserve, _ = seg_find_freezing_by_frozen(matlab_train)
        freeze_tr, _ = seg_find_freezing_by_frozen(train21)
        GT_tr, _ = seg_find_freezing_by_frozen(train11)
        
    (err12_conserve, mean_dist12_conserve, _)=supp_methods.freezing_metrics(np.asarray(freeze2_conserve),np.asarray(freeze_GT), 10)
    (err12_tr, mean_dist12_tr, _)=supp_methods.freezing_metrics(np.asarray(freeze_tr),np.asarray(GT_tr), 10)
    (err12_matlab, mean_dist12_matlab, _)=supp_methods.freezing_metrics(np.asarray(matlab_conserve),np.asarray(freeze_GT), 10)
    # print("CV Scores: ", pd.DataFrame(cv_scores))
    
    # lets see what feature we used
    #print("Features: ", clf.steps[1][1].f_labels)
    
    
    None
