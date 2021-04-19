#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 10:43:39 2021

@author: andrey
"""
from data_augmentation_lib import augment_data
from unite_datasets import unite_datasets
bacteria_files=["0_raw_dataset_384bact0freez31.hdf5"]
water_files=[]
bacteria_result_files=["_test_bact384_aug.hdf5"]
water_result_files=[]
united_file=[]
# number of times to repeat 
batches_num = 2
angles_num =5
### Let us Augment data first
### first augment bacteria files
for i in range(len(bacteria_files)):
    for j in range(batches_num):
        augment_data(bacteria_files[i], str(j)+bacteria_result_files, angles_num, 42, False)
for i in range(len(water_files)):
    for j in range(batches_num):
        augment_data(water_files[i], str(j)+water_result_files, angles_num, 42, False)
### now we will unite the datasets
list_to_unite = []
#list_to_unite.clear()
for i in range (len(bacteria_files)):
    list_to_unite+=[str(j)+bacteria_result_files[i] for j in range(batches_num)]
for i in range (len(water_files)):
    list_to_unite+=[str(j)+water_result_files[i] for j in range(batches_num)]
