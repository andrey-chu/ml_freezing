#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:00:59 CET 2020
This file implements HMM using pomegranade package HHM-routines
@author: andrey


"""
import h5py
import numpy as np
import supp_methods
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pomegranate import HiddenMarkovModel
from pomegranate import NormalDistribution
from pomegranate import MultivariateGaussianDistribution
import scipy
from seglearn.feature_functions import mean, median, abs_energy, std, skew, mean_crossings, minimum, maximum, mean_diff,\
     zero_crossing, var
# import pandas as pd
# from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import GridSearchCV
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import RepeatedKFold
# from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, cross_validate

# from sklearn.ensemble import RandomForestClassifier
from seglearn.pipe import Pype
from seglearn.transform import FeatureRep, SegmentX, SegmentXY 
# from sklearn.metrics import f1_score, make_scorer
# from seglearn.base import TS_Data
# from seglearn.transform import FunctionTransformer
# from seglearn.feature_functions import mean, median, abs_energy, std, skew, mean_crossings, minimum, maximum, mean_diff,\
# zero_crossing, var
# from supp_methods import seg_find_freezing_by_frozen
# from seqlearn.hmm import MultinomialHMM
# from seqlearn.perceptron import StructuredPerceptron
from supp_methods import seg_find_freezing_by_frozen
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
def my_lists_to_array(inp_list, dims, placeholder):
    out_array = np.ones(dims)*placeholder
    # import pdb; pdb.set_trace()
    for i in range(dims[0]):
        out_array[i,:inp_list[i].shape[0],:,:]=inp_list[i]
    return out_array
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
def inverse_transform1(array1, lengths):
    new_list = [None]*len(lengths)
    temp = 0
    # import pdb; pdb.set_trace()
    for i in range(len(lengths)):
        new_list[i] = array1[temp:temp+lengths[i]]
        temp+=lengths[i]
    return new_list
def inverse_transform2(array1, lengths):
    new_list = [None]*len(lengths)
    temp = 0
    # import pdb; pdb.set_trace()
    for i in range(len(lengths)):
        new_list[i] = array1[temp:temp+lengths[i], :]
        temp+=lengths[i]
    return new_list
def linear_f(a_for_size):
    size = len(a_for_size)
    return np.linspace(0,size,size)
def my_diff(param_to_diff):
    from scipy.signal import savgol_filter
    return  savgol_filter(np.gradient(param_to_diff), 5,3)
fun_dict ={
        "lin":linear_f,
        "grad":my_diff
        }

def add_features(prev_features, to_add):
    new_features = [None]*len(prev_features)
    for i in range(len(prev_features)):
        #length_add = prev_features[i].shape[0]
        it = prev_features[i].shape[1]
        add_feat_length = len(to_add)
        # create an empty array
        new_features[i]=np.empty((prev_features[i].shape[0],prev_features[i].shape[1]+add_feat_length))
        # add existing features
        new_features[i][:,:prev_features[i].shape[1]] = prev_features[i]
        for funs in to_add:
            funs_split = funs.split("_")
            if len(funs_split)==2:
                funa = fun_dict[funs_split[0]]
                funb = np.int(funs_split[1])
            else:
                funa = fun_dict[funs]
                funb = 0
            
            
            new_features[i][:,it]=funa(prev_features[i][:,funb])
            it+=1
    return new_features


#datadir = "/data/Freezing_samples/h5data_new/"
united_dataset = datadir + "united_raw_dataset_384freez31f2.hdf5"
add_dataset1 = datadir + "united_raw_dataset_384freez31f2_aug.hdf5"
change_labels_bool =1
conservative = 0
with h5py.File(united_dataset, 'r') as f, h5py.File(add_dataset1, 'r') as f1:
    d_images = f["Raw_data/images_dataset"]
    d_labels = f["Raw_data/labels_dataset"]
    d_labels_add = f1["Raw_data/labels_dataset"]
    d_features = f["Raw_data/features2_dataset"]
    d_features_add = f1["Raw_data/features2_dataset"]
    d_matlab = f["Raw_data/matlab_dataset"]
    d_matlab_add = f1["Raw_data/matlab_dataset"]
    if d_matlab_add.shape[0]==1:
        reshape_me = True
    else:
        reshape_me = False
    d_exclude=f["Raw_data/exclude_dataset"]
    d_exclude_add=f1["Raw_data/exclude_dataset"]
    d_shapes=f["Raw_data/shapes_dataset"]
    d_shapes_add=f1["Raw_data/shapes_dataset"]
    d_datasets=f["Raw_data/datasets_dataset"]
    d_substance=f["Raw_data/substance_dataset"]
    features_data = d_features[:]
    features_data_add = d_features_add[:]
    shapes = d_shapes[:]
    shapes_add = d_shapes_add[:]
    exclude = d_exclude[:]
    # import pdb; pdb.set_trace()
    exclude_add = d_exclude_add[:]
    dataset_im_shape = d_images.shape
    dataset_lb_shape = d_labels.shape
    labels = d_labels[:]
    labels_add = np.uint8(d_labels_add[:])
    matlab = d_matlab[:]
    matlab_add = d_matlab_add[:]
    datasets=d_datasets[:]
    substance = d_substance[:].flatten()
    
    total_number_wells = d_images.shape[0]
    
    support = supp_methods.create_2d_support(shapes, exclude, labels.shape)
    if reshape_me == True:
        matlab_add = np.reshape(matlab_add, (matlab_add.shape[1], matlab_add.shape[2]))
        labels_add = np.reshape(labels_add, (labels_add.shape[1], labels_add.shape[2]))
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
        new_labels_add = change_labels_preserve(labels_add,3)
        new_matlab = change_labels_preserve(matlab,3)
        new_matlab_add = change_labels_preserve(matlab_add,3)
    else:
        new_labels_bool = labels
        new_matlab = matlab
    # now let us turn all the data into a format suitable
    #chosen_features = [1,2,3, 5, 8, 9]
    chosen_features = [1,2]
    #chosen_features = [3]
    #chosen_features=[0,1,2,3,5, 6,7,8,9,13]
    new_features_seg = features2seg(features_data[:,chosen_features,:],shapes)
    new_features_add_seg = features2seg(features_data_add[:,chosen_features,:],shapes_add)
    #import pdb; pdb.set_trace()
    additional_features=['grad_0']
    new_features_seg1 = add_features(new_features_seg, additional_features)
    new_features_add_seg1 = add_features(new_features_add_seg, additional_features)
    
    new_labels_seg = labels2seg(new_labels,shapes)
    
    new_matlab_seg = labels2seg(new_matlab, shapes)
    new_labels_add_seg = labels2seg(new_labels_add,shapes_add)
    new_matlab_add_seg = labels2seg(new_matlab_add, shapes_add)    
    # now let us exclude the bad wells
    new_features_seg_included = [new_features_seg1[i] for i in np.nonzero(np.logical_not(exclude))[1]]
    new_labels_seg_included = [new_labels_seg[i] for i in np.nonzero(np.logical_not(exclude))[1]]
    new_features_add_seg_included = [new_features_add_seg1[i] for i in np.nonzero(np.logical_not(exclude_add))[1]]
    new_labels_add_seg_included = [new_labels_add_seg[i] for i in np.nonzero(np.logical_not(exclude_add))[1]]

    new_matlab_seg_included = [new_matlab_seg[i] for i in np.nonzero(np.logical_not(exclude))[1]]
    # datasets_included = datasets[np.nonzero(np.logical_not(exclude))[1]]
    # substance_included = substance[np.nonzero(np.logical_not(exclude))[1]]
    # for investigation
    # create a feature representation pipeline
    chosenwidth=6
    
    lengths = [new_features_seg_included[i].shape[0] for i in range(len(new_features_seg_included))]
    lengths_add = [new_features_add_seg_included[i].shape[0] for i in range(len(new_features_add_seg_included))]
    X_train, X_test, y_train, y_test, matlab_train, matlab_test, lengths_train, lengths_test = train_test_split(new_features_seg_included, new_labels_seg_included, new_matlab_seg_included, lengths, test_size=0.20, random_state=10)
    #X_train_new = np.vstack()
    train_summed=new_features_add_seg_included+X_train
    y_train_add = new_labels_add_seg_included + y_train

    fts = {'mean': mean, 'var': var, 'std': std, 'skew': skew, 'mnx': mean_crossings, 'minimum':minimum, 'maximum':maximum, \
           'mean_diff':mean_diff}
    feature_ext = FeatureRep(fts)
    lengths_add_train1 = lengths_add + lengths_train
    lengths_add_train = [i-chosenwidth+1 for i in lengths_add_train1]
    lengths_test1 = [i-chosenwidth+1 for i in lengths_test]
    
    tform =SegmentXY(width=chosenwidth,step=1)
    train_temp=[None]*len(train_summed)
    test_temp=[None]*len(X_test)
    train_X_list=[None]*len(train_summed)
    train_Y_list=[None]*len(train_summed)
    test_X_list=[None]*len(X_test)
    test_Y_list=[None]*len(X_test)
    for i in range(len(train_summed)):
        train_temp[i]= tform.fit_transform([train_summed[i]],[y_train_add[i]])
    
    for i in range(len(X_test)):   
        test_temp[i] = tform.fit_transform([X_test[i]],[y_test[i]])
    for i in range(len(train_summed)):
        train_X_list[i] = train_temp[i][0]
        train_Y_list[i] = list(train_temp[i][1])
    for i in range(len(X_test)):
        test_X_list[i] = test_temp[i][0]
        test_Y_list[i] = list(test_temp[i][1])
    dim1 = len(train_summed)
    dim2 = max(lengths_add_train)
    dim3 = chosenwidth
    dim4 = len(chosen_features)+len(additional_features)
    train_X_array=my_lists_to_array(train_X_list, (dim1, dim2, dim3, dim4), np.nan)
    # modelHMM = HiddenMarkovModel.from_samples(
            # MultivariateGaussianDistribution, 2, train_X_list, labels=train_Y_list, algorithm='labeled')
    states = 3
    prob_start = np.array([1, 0, 0])
    prob_ends = np.array([0,0,1])
    pre_pype1 = Pype([('seg',tform), ('features', feature_ext), ('scaler', StandardScaler())])
    pre_pype1.fit(train_summed, y_train_add)
    (x_prepared, y_prepared)=pre_pype1.transform(train_summed, y_train_add)
    
    x_prepared_reshaped = inverse_transform2(x_prepared, lengths_add_train)
    y_prepared_reshaped = inverse_transform1(y_prepared, lengths_add_train)
    
    # d1 = 
    # trans_mat = np.array([[0.5, 0.5, 0],
    #                       [0, 0.5, 0.5],
    #                       [0, 0, 0.5]])
    # 
    # d1 = NormalDistribution(1, 1)
    # d2 = NormalDistribution(-1, 0.75)
    # d3 = NormalDistribution(-1, 0.7)
    # states = [d1, d2, d3]
    # modelHMM1= HiddenMarkovModel.from_matrix(trans_mat, states, prob_start, prob_ends, merge = 'None')
    # modelHMM1.bake()
    # modelHMM1.fit(train_X_list)
    # tform =SegmentXY(width=chosenwidth,step=1)
    
    # train_summed_seg,y_train_summed_seg,_ = tform.fit_transform(train_summed,y_train_add)
    # test_summed_seg,y_test_summed_seg,_ = tform.fit_transform(X_test,y_test)
    # train_summed_seg1=train_summed_seg.reshape(-1,chosenwidth*(len(chosen_features)+len(additional_features)))
    # test_summed_seg1=test_summed_seg.reshape(-1,chosenwidth*(len(chosen_features)++len(additional_features)))
    # #X_train_add = np.vstack(new_features_add_seg_included+X_train)
    
    
    # # import pdb; pdb.set_trace()
    # #X_test_new = np.vstack(X_test)
    # scaler = StandardScaler().fit(train_summed_seg1)
    # X_train_sc = scaler.transform(train_summed_seg1)
    # X_test_sc = scaler.transform(test_summed_seg1)
    # lengths_add_train1 = lengths_add + lengths_train
    # lengths_add_train = [i-chosenwidth+1 for i in lengths_add_train1]
    # lengths_test1 = [i-chosenwidth+1 for i in lengths_test]
    # y_train1 = [np.reshape(y_train_add[i], (-1,1)) for i in range(len(y_train_add))]
    # y_test1 = [np.reshape(y_test[i], (-1,1)) for i in range(len(y_test))]
    # y_train_new = np.vstack(y_train1)
    # y_test_new = np.vstack(y_test1)
    # y_train_new1=y_train_summed_seg.reshape(-1,1)
    # y_test_new1=y_test_summed_seg.reshape(-1,1)
    # import pdb; pdb.set_trace()
    # # clf = StructuredPerceptron(decode="bestfirst", lr_exponent=0.1, max_iter=1000, random_state=None, trans_features=False, verbose=5)
    # clf1 = StructuredPerceptron(decode="bestfirst", lr_exponent=1, max_iter=1000, random_state=None, trans_features=False, verbose=5)
    # clf2 = MultinomialHMM(decode="bestfirst", alpha=0.1)
    # clf.fit(X_train_sc, y_train_new, lengths_train)
    
    # X_train_sc1, y_train_sc1, sample_weight_new=tform.fit_transform(X_train_sc, y_train_new)
    # import pdb; pdb.set_trace()
    
    # clf1.fit(X_train_sc, y_train_new1, lengths_add_train)
    # # clf2.fit(X_train_sc, y_train_new1, lengths_add_train)
    # # y_pred = clf.predict(np.float64(X_test_sc), lengths_test)
    # y_pred1 = clf1.predict(np.float64(X_test_sc), lengths_test1)
    # # y_pred2 = clf2.predict(np.float64(X_test_sc), lengths_test1)
    # Y = inverse_transform1(y_pred1, lengths_test1)
    # # Y2 = inverse_transform1(y_pred2, lengths_test1)
    # test_fp, _=seg_find_freezing_by_frozen(Y)
    # # test_fp2, _=seg_find_freezing_by_frozen(Y2)
    # matlab_res, _ = seg_find_freezing_by_frozen(matlab_test)
    # ground_truth, _=seg_find_freezing_by_frozen(y_test)
    # (err, mean_dist, _)=supp_methods.freezing_metrics(np.asarray(test_fp),np.asarray(ground_truth), 10)
    # (err_mat, mean_dist_mat, _)=supp_methods.freezing_metrics(np.asarray(matlab_res),np.asarray(ground_truth), 10)
    None
