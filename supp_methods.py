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
    it returns a tuple of (precision, recall, F1, weighted_distance_from_fr_point)
    and also prints it out if printinfo is given true
    """
    import numpy as np
    import sklearn.metrics as skl_met
    labels=np.union1d(np.unique(gt_labels),np.unique(estimated_labels))
    # confusion matrix
    confusion_matrix_skl=skl_met.confusion_matrix(estimated_labels, gt_labels)
    confusion_matrix_mine1 = confusion_matrix_mine(estimated_labels,gt_labels, labels)
    class_report_dict = skl_met.classification_report(estimated_labels, gt_labels,labels)
    precision_calc = class_report_dict["weighted avg"]["precision"]
    recall_calc = class_report_dict["weighted avg"]["recall"]
    F1_calc = class_report_dict["weighted avg"]["f1-score"]
    if printinfo:
        print(confusion_matrix_skl)
        print(confusion_matrix_mine1)
    return (precision_calc, recall_calc, F1_calc)
    
def confusion_matrix_mine(estimated_labels, gt_labels, labels):
    import numpy as np
    matrix = np.zeros
    for i in range(len(labels)):
        for j in range(len(labels)):
            matrix[i,j] = (estimated_labels==labels[i])&(gt_labels==labels[j])

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
    
