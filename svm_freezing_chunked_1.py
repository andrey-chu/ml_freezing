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

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import learning_curve
import platform
if platform.node()=='choo-desktop':
    from branch_init_choo import datadir
elif platform.node()=='andrey-cfin':
    from branch_init_cfin import datadir
from sklearn.model_selection import train_test_split, cross_validate

chunked_united_dataset = datadir + "united_chunked_dataset_96freez31.hdf5"

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
    with_ones =1 # if 1 then ones will remain
    support = supp_methods.create_2d_support_chunked(shapes, exclude, labels.shape)
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
    #(training_data,cv_data,test_data,train_set_wells,_,test_set_wells) = supp_methods.random_divide_samples(support, exclude, 0.6,0.2)
    features_data = d_features[:]
    features_data_included = features_data[:,:,:,np.nonzero(np.logical_not(exclude))[1]]
    labels_included = labels[:,np.nonzero(np.logical_not(exclude))[1]]
    matlab_included = matlab[:,np.nonzero(np.logical_not(exclude))[1]]
    features_data_included_resh = np.swapaxes(features_data_included, 0,3)
    labels_included_resh = np.transpose(labels_included)
    matlab_included_resh = np.transpose(matlab_included)
    conc_feat = [0,1,2,3,5, 6,7,8,9,13]#3,4,6
    features_data_included_resh_conc = features_data_included_resh[:,conc_feat,:,:]
    features_data_included_resh_conc1 = np.swapaxes(np.swapaxes(features_data_included_resh_conc, 1,3), 2,3)
    X_in=np.reshape(features_data_included_resh_conc1,(features_data_included_resh_conc1.shape[0],features_data_included_resh_conc1.shape[1],-1))
    X_train, X_test, y_train, y_test, matlab_train, matlab_test = train_test_split\
        (X_in, labels_included_resh, matlab_included_resh, test_size=0.10,random_state=10)
    # training_data_2d = np.nonzero(training_data.reshape(support.shape))
    # cv_data_2d = np.nonzero(cv_data.reshape(support.shape))
    # test_data_2d = np.nonzero(test_data.reshape(support.shape))
    X_train1 = np.reshape(X_train, (-1,X_train.shape[2]))
    X_test1 = np.reshape(X_test, (-1,X_test.shape[2]))
    y_train1 = y_train.flatten()
    y_test1 = y_test.flatten()
    matlab_train1 = matlab_train.flatten()
    matlab_test1 = matlab_test.flatten()
    scaler = StandardScaler().fit(X_train1)
    X_train_sc = scaler.transform(X_train1)
    X_test_sc = scaler.transform(X_test1)
    y_train_wout_1 = y_train1.copy()
    # next line removes 1's
    if with_ones == 1:
        y_train_wout_1[y_train_wout_1==1]=2
    y_test_wout_1 = y_test1.copy()
    # next line removes 1's
    if with_ones == 1:
        y_test_wout_1[y_test_wout_1==1]=2


    
    
    C_chosen = 77.42#12.915#77.42#grid1.best_params_["C"]
    gamma_chosen = 0.05994#grid1.best_params_["gamma"]
    n_estimators = 20
    clf3 = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', gamma=gamma_chosen, C=C_chosen, probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, warm_start=True, n_estimators=n_estimators, n_jobs=6, verbose=10))
    #let us remove all the nans in the end of the sequences
    mask100 = y_train_wout_1!=100
    X_train_sc_100 = X_train_sc[mask100,:]
    y_train_wout_100 = y_train_wout_1[mask100]
    
    matlab_train_wout1 = matlab_train1.copy()
    # next line removes 1's
    if with_ones == 1:
        matlab_train_wout1[matlab_train_wout1==1]=2
    matlab_test_wout1 = matlab_test1.copy()
    if with_ones == 1:
         # next line removes 1's
        matlab_test_wout1[matlab_test_wout1==1]=2
   
    matlab_test_nan = np.float_(matlab_test_wout1.copy())
    matlab_test_nan[matlab_test_wout1==100]=np.nan

    matlab_train_nan = np.float_(matlab_train_wout1.copy())
    matlab_train_nan[matlab_train_wout1==100]=np.nan
    
    
    test_mask100 = y_test_wout_1!=100
    y_test_wout100 = y_test_wout_1[test_mask100]
    X_test_sc_100 = X_test_sc[test_mask100,:]
    ## CV
    # C_range = np.logspace(-2, 5, 10)
    # gamma_range = np.logspace(-4, 1, 10)
    # param_grid = dict(gamma=gamma_range, C=C_range)
    # cv = StratifiedShuffleSplit(n_splits=10, train_size=0.01, test_size=0.2, random_state=42)
    # grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv,n_jobs=6, verbose=10)
    # grid.fit(X_train_sc_100, y_train_wout_100)

    # print("The best parameters are %s with a score of %0.2f"
    #   % (grid.best_params_, grid.best_score_))
    ## end of CV
    
    
    # Learning curve
    # train_sizes, train_scores, test_scores = learning_curve(clf3, X_train_sc_100, y_train_wout_100,\
    #                                                         cv=10, n_jobs=6, verbose=10)#, train_sizes=train_sizes
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    # plt.figure()
    # plt.title('learning curve')
    # plt.grid()
    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                 train_scores_mean + train_scores_std, alpha=0.1,
    #                 color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
    #         label="Training score")
    # plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
    #         label="Cross-validation score")

    # plt.legend(loc="best")
    # end of learning curve
    
    # everything is ready, let's fit
    clf3.fit(X_train_sc_100, y_train_wout_100)
    score = clf3.score(X_train_sc_100, y_train_wout_100)
    Predicted_test3 = clf3.predict(X_test_sc_100)
    Predicted_train3 = clf3.predict(X_train_sc_100)
    
    ## now let's add the nans and reshape back
    test_predicted = np.empty(test_mask100.shape)
    test_predicted[test_mask100]=Predicted_test3
    test_predicted[np.logical_not(test_mask100)]=np.nan
    test_predicted_reshaped = np.reshape(test_predicted, y_test.shape)
    
    train_predicted = np.empty(mask100.shape)
    train_predicted[mask100]=Predicted_train3
    train_predicted[np.logical_not(mask100)]=np.nan
    train_predicted_reshaped = np.reshape(train_predicted, y_train.shape)
    
    GT = np.empty(test_mask100.shape)
    GT[test_mask100]=y_test_wout100
    GT[np.logical_not(test_mask100)]=np.nan
    GT_reshaped=np.reshape(GT, y_test.shape)

    GT_tr = np.empty(mask100.shape)
    GT_tr[mask100]=y_train_wout_100
    GT_tr[np.logical_not(mask100)]=np.nan
    GT_tr_reshaped=np.reshape(GT_tr, y_train.shape)
    
    Matlab_reshaped = np.reshape(matlab_test_nan, y_test.shape)
    Matlab_tr_reshaped = np.reshape(matlab_train_nan, y_train.shape)
    
    (test_freeze1, test_freeze2) = supp_methods.find_freezing_by_frozen(test_predicted_reshaped.T, np.int_(np.ones((1,y_test.shape[0]))))
    (train_freeze1, train_freeze2) = supp_methods.find_freezing_by_frozen(train_predicted_reshaped.T,np.int_(np.ones((1,y_train.shape[0]))))
    (GT_freeze1, GT_freeze2) = supp_methods.find_freezing_by_frozen(GT_reshaped.T, np.int_(np.ones((1,y_test.shape[0]))))
    (GT_tr_freeze1, GT_tr_freeze2) = supp_methods.find_freezing_by_frozen(GT_tr_reshaped.T,np.int_(np.ones((1,y_train.shape[0]))))    
    
    (Matlab_freeze1, Matlab_freeze2) = supp_methods.find_freezing_by_frozen(Matlab_reshaped.T, np.int_(np.ones((1,y_test.shape[0]))))
    (Matlab_tr_freeze1, Matlab_tr_freeze2) = supp_methods.find_freezing_by_frozen(Matlab_tr_reshaped.T,np.int_(np.ones((1,y_train.shape[0]))))        
#    (aa4, bb4) = supp_methods.find_freezing_by_frozen(Predicted4,test_set_wells)
    # 3,4,6,7,8,9,14
#    pr1_fr = supp_methods.freezing_est_statistic(bb1, test_set_wells)
#    pr2_fr = supp_methods.freezing_est_statistic(bb2, test_set_wells)
    pr3_fr = supp_methods.freezing_est_statistic(test_freeze2, np.int_(np.ones((1,y_test.shape[0]))))
    pr3_tr_fr = supp_methods.freezing_est_statistic(train_freeze2, np.int_(np.ones((1,y_train.shape[0]))))
#    pr4_fr = supp_methods.freezing_est_statistic(bb4, test_set_wells)
    gt_fr = supp_methods.freezing_est_statistic(GT_freeze2, np.int_(np.ones((1,y_test.shape[0]))))
    gt_tr_fr = supp_methods.freezing_est_statistic(GT_tr_freeze2, np.int_(np.ones((1,y_train.shape[0]))))
    matlab_fr = supp_methods.freezing_est_statistic(Matlab_freeze2, np.int_(np.ones((1,y_test.shape[0]))))
    matlab_tr_fr = supp_methods.freezing_est_statistic(Matlab_tr_freeze2, np.int_(np.ones((1,y_train.shape[0]))))
#    (err1, mean_dist1, _)=supp_methods.freezing_metrics(pr1_fr,gt_fr, 10)
#    (err2, mean_dist2, _)=supp_methods.freezing_metrics(pr2_fr,gt_fr, 10)
    (err3, mean_dist3, _)=supp_methods.freezing_metrics(pr3_fr,gt_fr, 10)
    (err3_tr, mean_dist3_tr, _)=supp_methods.freezing_metrics(pr3_tr_fr,gt_tr_fr, 10)
#    (err4, mean_dist4, _)=supp_methods.freezing_metrics(pr4_fr,gt_fr, 10)
    (err_mat, mean_dist_mat, _)=supp_methods.freezing_metrics(matlab_fr,gt_fr, 10)
    
    
    None
    