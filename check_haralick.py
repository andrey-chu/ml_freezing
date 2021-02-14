#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:15:04 2020

@author: andrey
"""

import platform
import h5py
import numpy as np
#import mahotas as mt

from matplotlib import pyplot as plt
if platform.node()=='choo-desktop':
    from branch_init_choo import datadir
elif platform.node()=='andrey-cfin':
    from branch_init_cfin import datadir
elif platform.node()=='andrey-workbook':
    from branch_init_laptop import datadir
    
dataset_to_read1 = datadir+'0_raw_dataset_384bact0freez31f2.hdf5'
dataset_to_read2 = datadir+'0_raw_dataset_384bact0freez31e2_aug1.hdf5'
plt.close('all')
dataset_to_read3 = datadir+'united_raw_dataset_384freez31f2.hdf5'
with h5py.File(dataset_to_read1, "r", libver="latest") as f1, h5py.File(dataset_to_read2, "r", libver="latest") as f2:
    features_d1 = f1['Raw_data/features2_dataset']
    features_d11 = f1['Raw_data/features_dataset']
    features_d2 = f2['Raw_data/features2_dataset']
    labels_d1 =  f1['Raw_data/labels_dataset']
    labels_d2 =  f2['Raw_data/labels_dataset']
    print(features_d1.shape)
    length_feat = features_d1.shape[0]
    wells_num = features_d1.shape[2]
    wells = np.random.randint(0,wells_num, 30)
    #for feature_num in range(12):
    feature_num =6
    for well_num in (wells):
        angle = 0
        well = well_num
        before_rotation = features_d1[:,feature_num,well]
        after_rotation = features_d2[:,feature_num,well+angle*wells_num]
       
        labels1=np.squeeze(labels_d1)[:,well]
        labels2 =np.squeeze(labels_d2)[:,well+angle*wells_num]
        fp1=np.where(labels1==1)[0][0]
        fp2=np.where(labels2==1)[0][0]
        #import pdb; pdb.set_trace()
        print(features_d1.shape)
        print(features_d2.shape)
        plt.figure()
        plt.title('Haralick comparison')
        plt.grid()
        plt.plot(range(length_feat), before_rotation, '-', color="g",
                 label="Feature {0} before rotation".format(feature_num))
        plt.plot(range(length_feat), after_rotation, '-', color="r",
                 label="Feature {0} after rotation".format(feature_num))
        plt.axvline(x=fp1,linestyle=':',color="g",label="FP before rotation")
        plt.axvline(x=fp2,linestyle=':',color="r",label="FP after rotation")
        
        plt.legend(loc="best")
        plt.show()
        
    