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

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

def features_pad(features, shapes):
        padded_features = features.copy()
        for i in range(features.shape[2]):
            padded_features[int(shapes[i,1]):,:,i]=padded_features[int(shapes[i,1]-1),:,i]
        return padded_features
    
def labels_pad(labels, shapes):
       padded_labels = labels.copy()
       for i in range(labels.shape[1]):
            padded_labels[int(shapes[i,1]):,i]=padded_labels[int(shapes[i,1]-1),i]
       return padded_labels


datadir = "/data/Freezing_samples/h5data_new/"
united_dataset = datadir + "united_raw_dataset_384freez31.hdf5"

with h5py.File(united_dataset, 'r') as f:
    d_images = f["Raw_data/images_dataset"]
    d_labels = f["Raw_data/labels_dataset"]
    d_features = f["Raw_data/features_dataset"]
    d_matlab = f["Raw_data/matlab_dataset"]
    d_exclude=f["Raw_data/exclude_dataset"]
    d_shapes=f["Raw_data/shapes_dataset"]
    features_data = d_features[:]
    shapes = d_shapes[:]
    exclude = d_exclude[:]
    dataset_im_shape = d_images.shape
    dataset_lb_shape = d_labels.shape
    labels = d_labels[:]
    matlab = d_matlab[:]
    total_number_wells = d_images.shape[0]
    
    support = supp_methods.create_2d_support(shapes, exclude, labels.shape)
    #total_number_chunks = support.shape[0]*support.shape[1]
    support_1col = support.reshape(1,-1).T
    labels_1col = labels.reshape(1,-1).T
    matlab_1col = matlab.reshape(1,-1).T
    included_ind_1_col=np.nonzero(support_1col)
    labels_1col_included = labels_1col[included_ind_1_col[0],:]
    matlab_1col_included = matlab_1col[included_ind_1_col[0],:]
    included_number_wells = np.sum(np.logical_not(exclude))
    ###included_number_chunks= int(np.sum(support.flatten()))
    metric1=supp_methods.calc_eval_metric(matlab_1col_included, labels_1col_included, True, dataset_lb_shape)
    metric2 = supp_methods.calc_add_metric(matlab, labels, exclude)
    #(a,b,c) = supp_methods.random_divide_sample_chunks(included_number_chunks, 0.6,0.2,0.2)
    #(training_data,cv_data,test_data,train_set_wells,_,test_set_wells) = supp_methods.random_divide_samples(support, exclude, 0.6,0.2)
    #### we will pad now both features and labels
    
    features_padded=features_pad(features_data, shapes)
    labels_padded = labels_pad(labels, shapes)
    matlab_padded = labels_pad(matlab, shapes)
    features_included = features_padded[:,:,np.nonzero(np.logical_not(exclude))[1]]
    labels_included = labels_padded[:,np.nonzero(np.logical_not(exclude))[1]]
    matlab_included = matlab_padded[:,np.nonzero(np.logical_not(exclude))[1]]
#    conc_feat = [0,1,2,3,5, 6,7,8,9,13]#3,4,6
#    conc_feat = [2,3,5]
    conc_feat = [2]
    features_reshaped = np.swapaxes(features_included[:,conc_feat,:], 0,2)
    features_reshaped2 = features_reshaped.reshape(features_reshaped.shape[0],-1)
    labels_reshaped= np.transpose(labels_included)
    matlab_reshaped = np.transpose(matlab_included)
    f_points = np.argmax(labels_reshaped==1, axis=1)
    matlab_f_points = np.argmax(matlab_reshaped==1, axis=1)
    
    
    # now y is a freezing point, for the purpose of regression test
    # X is an array of features (all 14 Haralick features)
    X_train, X_test, y_train, y_test, matlab_f_points_train, matlab_f_points_test =\
    train_test_split(features_reshaped2, f_points, matlab_f_points, test_size=0.2)
    
    

    #conc_feat = [2,3,5, 6,7,13]#3,4,6
    
    
    
    
    
    # all the data intact, apparently cv algorithms in sk-learn are smart 
    # enough to do splitting for cross validation by themselves
    #X_all = np.vstack((X,X_cv))# maybe we need to add test here, but I am actually not sure
    
    #### next line removes freezing point for classification purposes
    ###### (because the class is very rare and may interfer with the results)

    #############################
    # let us scale the data
    scaler = StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    
    
    
    # searching for the values of C and gamma for RBF kernel
    C_range = np.logspace(-2, 5, 10)
    gamma_range = np.logspace(-4, 1, 10)
    epsilon_range = np.logspace(-4, 5, 10)
    
#    C_range = np.linspace(2700, 2700, 1)
#    gamma_range = np.logspace(-10, -4, 20)
#    epsilon_range = np.logspace(-10, -4, 20)
    param_grid = dict(gamma=gamma_range, C=C_range, epsilon=epsilon_range)
    cv = ShuffleSplit(n_splits=10, train_size=0.2, test_size=0.8, random_state=41)
#    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv,n_jobs=6, verbose=10)
#    grid.fit(X_all, Y_all.flatten())
    grid1= GridSearchCV(SVR(), param_grid=param_grid, cv=cv,n_jobs=6, verbose=10)
    grid1.fit(X_train, y_train)
    print("The best parameters are %s with a score of %0.2f"
      % (grid1.best_params_, grid1.best_score_))
    #C_chosen = 12.915#77.42#grid1.best_params_["C"]
    #gamma_chosen = 0.05994#grid1.best_params_["gamma"]
#    svm = SVC(kernel='rbf', random_state=0, gamma=gamma_chosen, C=C_chosen,\
#              n_jobs=5, verbose=10)
    #gamma_chosen = 0.0001
    gamma_chosen = grid1.best_params_["gamma"]
    epsilon_chosen = grid1.best_params_["epsilon"]
    C_chosen = grid1.best_params_["C"]#C_chosen = 2700
    n_estimators = 20
#    clf1 = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', gamma=grid1.best_params_["gamma"], C=grid1.best_params_["C"], probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=6, verbose=10))
#    clf2 = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', gamma=grid.best_params_["gamma"], C=grid.best_params_["C"], probability=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=6, verbose=10))
    clf3 = SVR(kernel='rbf', gamma=gamma_chosen, epsilon=epsilon_chosen, C=C_chosen)
#    clf4 = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', gamma=grid.best_params_["gamma"], C=grid.best_params_["C"], probability=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators, warm_start=True, n_jobs=6, verbose=10))
#    clf1.fit(X_all_conc, Y_all.flatten())
#    clf2.fit(X_all_conc, Y_all.flatten())
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    cv1 = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
#    Y_flattened=Y_all_wout_1.flatten()
    train_sizes, train_scores, test_scores = learning_curve(clf3, X_train, y_train,\
                                                            cv=10, n_jobs=6)#, train_sizes=train_sizes
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure()
    plt.title('Learning curve')
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
     
    plt.legend(loc="best")
    clf3.fit(X_train, y_train)
#    clf4.fit(X_all_conc, Y_all.flatten())
#    Predicted_test1 = clf1.predict(X_test_conc)
#    Predicted_test2 = clf2.predict(X_test_conc)
    Predicted_test3 = clf3.predict(X_test)
    Predicted_train3 = clf3.predict(X_train)
#    Predicted_test4 = clf4.predict(X_test_conc)
#    supp_methods.calc_eval_metric(Predicted_test1, Y_test, True, (2287,48))
#    Predicted1 = supp_methods.transform_back(support.shape, test_data_2d, Predicted_test1)
#    Predicted2 = supp_methods.transform_back(support.shape, test_data_2d, Predicted_test2)
#    Predicted3 = supp_methods.transform_back(support.shape, test_data_2d, Predicted_test3)
#    Train3 = supp_methods.transform_back(support.shape, training_data_2d, Predicted_train3)
#    Predicted4 = supp_methods.transform_back(support.shape, test_data_2d, Predicted_test4)
#    GT1 = supp_methods.transform_back(support.shape, test_data_2d, Y_test.flatten())
#    GT1_tr = supp_methods.transform_back(support.shape, training_data_2d, Y.flatten())
#    Matlab1 = supp_methods.transform_back(support.shape, test_data_2d, matlab_test.flatten())
    # let us try to choose particular features
#    (aa1, bb1) = supp_methods.find_freezing_by_frozen(Predicted1,test_set_wells)
#    (aa2, bb2) = supp_methods.find_freezing_by_frozen(Predicted2,test_set_wells)
#    (aa3, bb3) = supp_methods.find_freezing_by_frozen(Predicted3,test_set_wells)
#    (aa3_tr, bb3_tr) = supp_methods.find_freezing_by_frozen(Train3,train_set_wells)
#    (aa4, bb4) = supp_methods.find_freezing_by_frozen(Predicted4,test_set_wells)
    # 3,4,6,7,8,9,14
#    pr1_fr = supp_methods.freezing_est_statistic(bb1, test_set_wells)
#    pr2_fr = supp_methods.freezing_est_statistic(bb2, test_set_wells)
#    pr3_fr = supp_methods.freezing_est_statistic(bb3, test_set_wells)
#    pr3_tr_fr = supp_methods.freezing_est_statistic(bb3_tr, train_set_wells)
#    pr4_fr = supp_methods.freezing_est_statistic(bb4, test_set_wells)
#    gt_fr = supp_methods.freezing_est_statistic(GT1, test_set_wells)
#    gt_tr_fr = supp_methods.freezing_est_statistic(GT1_tr, train_set_wells)
#    matlab_fr = supp_methods.freezing_est_statistic(Matlab1, test_set_wells)
#    (err1, mean_dist1, _)=supp_methods.freezing_metrics(pr1_fr,gt_fr, 10)
#    (err2, mean_dist2, _)=supp_methods.freezing_metrics(pr2_fr,gt_fr, 10)
    (err3, mean_dist3, _)=supp_methods.freezing_metrics(Predicted_test3,y_test, 10)
    (err3_tr, mean_dist3_tr, _)=supp_methods.freezing_metrics(Predicted_train3,y_train, 10)
#    (err4, mean_dist4, _)=supp_methods.freezing_metrics(pr4_fr,gt_fr, 10)
    (err_mat, mean_dist_mat, _)=supp_methods.freezing_metrics(matlab_f_points_test, y_test, 10)
    None
