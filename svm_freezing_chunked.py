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
from branch_init import datadir

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
    (training_data,cv_data,test_data,train_set_wells,_,test_set_wells) = supp_methods.random_divide_samples(support, exclude, 0.6,0.2)
    training_data_2d = np.nonzero(training_data.reshape(support.shape))
    cv_data_2d = np.nonzero(cv_data.reshape(support.shape))
    test_data_2d = np.nonzero(test_data.reshape(support.shape))
    features_data_training = d_features[:]
    features_data_training = features_data_training[training_data_2d[0],:,:,training_data_2d[1]]
    features_data_cv = d_features[:]
    features_data_cv = features_data_cv[cv_data_2d[0],:,:,cv_data_2d[1]]
    features_data_test = d_features[:]
    features_data_test = features_data_test[test_data_2d[0],:,:,test_data_2d[1]]
    X = features_data_training.reshape(features_data_training.shape[0],-1)
    #conc_feat = [2,3,5, 6,7,13]#3,4,6
    conc_feat = [0,1,2,3,5, 6,7,8,9,13]#3,4,6
    
    features_data_training_conc = features_data_training[:,conc_feat,:]
    features_data_cv_conc = features_data_cv[:,conc_feat,:]
    features_data_test_conc = features_data_test[:,conc_feat,:]
    X_conc = features_data_training_conc.reshape(features_data_training_conc.shape[0],-1)
    X_cv_conc = features_data_cv_conc.reshape(features_data_cv_conc.shape[0],-1)
    X_cv = features_data_cv.reshape(features_data_cv.shape[0],-1)
    X_test = features_data_test.reshape(features_data_test.shape[0],-1)
    X_test_conc = features_data_test_conc.reshape(features_data_test_conc.shape[0],-1)
    Y = labels_1col[np.bool_(training_data)].reshape(1,-1).T
    Y_cv = labels_1col[np.bool_(cv_data)].reshape(1,-1).T
    Y_test = labels_1col[np.bool_(test_data)].reshape(1,-1).T
    matlab_test = matlab_1col[np.bool_(test_data)].reshape(1,-1).T
    X_all = np.vstack((X,X_cv))
    
    X_all_conc = np.vstack((X_conc,X_cv_conc))
    Y_all = np.vstack((Y,Y_cv))
    # all the data intact, apparently cv algorithms in sk-learn are smart 
    # enough to do splitting for cross validation by themselves
    X_all = np.vstack((X,X_cv))# maybe we need to add test here, but I am actually not sure
    Y_all = np.vstack((Y,Y_cv))
    #### next line removes freezing point for classification purposes
    ###### (because the class is very rare and may interfer with the results)
    Y_all_wout_1 = Y_all.copy()
    Y_all_wout_1[Y_all_wout_1==1]=2
    #############################
    # let us scale the data
    scaler = StandardScaler().fit(X_all)
    scaler_conc = StandardScaler().fit(X_all_conc)
    X_all=scaler.transform(X_all)
    X_all_conc=scaler_conc.transform(X_all_conc)
    X_conc=scaler_conc.transform(X_conc)
    X_test = scaler.transform(X_test)
    X_test_conc = scaler_conc.transform(X_test_conc)
    
    # searching for the values of C and gamma for RBF kernel
#    C_range = np.logspace(-2, 5, 10)
#    gamma_range = np.logspace(-4, 1, 10)
#    param_grid = dict(gamma=gamma_range, C=C_range)
#    cv = StratifiedShuffleSplit(n_splits=10, train_size=0.01, test_size=0.2, random_state=42)
#    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv,n_jobs=6, verbose=10)
#    grid.fit(X_all, Y_all.flatten())
#    grid1= GridSearchCV(SVC(class_weight='balanced'), param_grid=param_grid, cv=cv,n_jobs=6, verbose=10)
#    grid1.fit(X_all, Y_all.flatten())
#    print("The best parameters are %s with a score of %0.2f"
#      % (grid.best_params_, grid.best_score_))
    C_chosen = 12.915#77.42#grid1.best_params_["C"]
    gamma_chosen = 0.05994#grid1.best_params_["gamma"]
#    svm = SVC(kernel='rbf', random_state=0, gamma=gamma_chosen, C=C_chosen,\
#              n_jobs=5, verbose=10)
    n_estimators = 20
#    clf1 = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', gamma=grid1.best_params_["gamma"], C=grid1.best_params_["C"], probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=6, verbose=10))
#    clf2 = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', gamma=grid.best_params_["gamma"], C=grid.best_params_["C"], probability=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=6, verbose=10))
    clf3 = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', gamma=gamma_chosen, C=C_chosen, probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, warm_start=True, n_estimators=n_estimators, n_jobs=6, verbose=10))
#    clf4 = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', gamma=grid.best_params_["gamma"], C=grid.best_params_["C"], probability=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators, warm_start=True, n_jobs=6, verbose=10))
#    clf1.fit(X_all_conc, Y_all.flatten())
#    clf2.fit(X_all_conc, Y_all.flatten())
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    cv1 = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
    Y_flattened=Y_all_wout_1.flatten()
#    train_sizes, train_scores, test_scores = learning_curve(clf3, X_all_conc, Y_flattened,\
#                                                            cv=10, n_jobs=6)#, train_sizes=train_sizes
#    train_scores_mean = np.mean(train_scores, axis=1)
#    train_scores_std = np.std(train_scores, axis=1)
#    test_scores_mean = np.mean(test_scores, axis=1)
#    test_scores_std = np.std(test_scores, axis=1)
#    plt.figure()
#    plt.title('Pup')
#    plt.grid()
#    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                     train_scores_mean + train_scores_std, alpha=0.1,
#                     color="r")
#    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
#    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#             label="Training score")
#    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#             label="Cross-validation score")
#
#    plt.legend(loc="best")
    clf3.fit(X_all_conc, Y_all_wout_1.flatten())
#    clf4.fit(X_all_conc, Y_all.flatten())
#    Predicted_test1 = clf1.predict(X_test_conc)
#    Predicted_test2 = clf2.predict(X_test_conc)
    Predicted_test3 = clf3.predict(X_test_conc)
    Predicted_train3 = clf3.predict(X_conc)
#    Predicted_test4 = clf4.predict(X_test_conc)
#    supp_methods.calc_eval_metric(Predicted_test1, Y_test, True, (2287,48))
#    Predicted1 = supp_methods.transform_back(support.shape, test_data_2d, Predicted_test1)
#    Predicted2 = supp_methods.transform_back(support.shape, test_data_2d, Predicted_test2)
    Predicted3 = supp_methods.transform_back(support.shape, test_data_2d, Predicted_test3)
    Train3 = supp_methods.transform_back(support.shape, training_data_2d, Predicted_train3)
#    Predicted4 = supp_methods.transform_back(support.shape, test_data_2d, Predicted_test4)
    GT1 = supp_methods.transform_back(support.shape, test_data_2d, Y_test.flatten())
    GT1_tr = supp_methods.transform_back(support.shape, training_data_2d, Y.flatten())
    Matlab1 = supp_methods.transform_back(support.shape, test_data_2d, matlab_test.flatten())
    # let us try to choose particular features
#    (aa1, bb1) = supp_methods.find_freezing_by_frozen(Predicted1,test_set_wells)
#    (aa2, bb2) = supp_methods.find_freezing_by_frozen(Predicted2,test_set_wells)
    (aa3, bb3) = supp_methods.find_freezing_by_frozen(Predicted3,test_set_wells)
    (aa3_tr, bb3_tr) = supp_methods.find_freezing_by_frozen(Train3,train_set_wells)
#    (aa4, bb4) = supp_methods.find_freezing_by_frozen(Predicted4,test_set_wells)
    # 3,4,6,7,8,9,14
#    pr1_fr = supp_methods.freezing_est_statistic(bb1, test_set_wells)
#    pr2_fr = supp_methods.freezing_est_statistic(bb2, test_set_wells)
    pr3_fr = supp_methods.freezing_est_statistic(bb3, test_set_wells)
    pr3_tr_fr = supp_methods.freezing_est_statistic(bb3_tr, train_set_wells)
#    pr4_fr = supp_methods.freezing_est_statistic(bb4, test_set_wells)
    gt_fr = supp_methods.freezing_est_statistic(GT1, test_set_wells)
    gt_tr_fr = supp_methods.freezing_est_statistic(GT1_tr, train_set_wells)
    matlab_fr = supp_methods.freezing_est_statistic(Matlab1, test_set_wells)
#    (err1, mean_dist1, _)=supp_methods.freezing_metrics(pr1_fr,gt_fr, 10)
#    (err2, mean_dist2, _)=supp_methods.freezing_metrics(pr2_fr,gt_fr, 10)
    (err3, mean_dist3, _)=supp_methods.freezing_metrics(pr3_fr,gt_fr, 10)
    (err3_tr, mean_dist3_tr, _)=supp_methods.freezing_metrics(pr3_tr_fr,gt_tr_fr, 10)
#    (err4, mean_dist4, _)=supp_methods.freezing_metrics(pr4_fr,gt_fr, 10)
    (err_mat, mean_dist_mat, _)=supp_methods.freezing_metrics(matlab_fr,gt_fr, 10)
    None
    