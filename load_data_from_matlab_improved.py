#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:41:31 2019
This module loads the needed data from matlab files and saves them in h5py format

@author: andrey
"""
data_384_dirlist = ['/data/Freezing_samples/Test384Bakterie_1/','/data/Freezing_samples/Test384Vand_1/']
data_96_dirlist = ['/data/Freezing_samples/Test96Bakterie_1/', '/data/Freezing_samples/Test96Bakterie_2/', '/data/Freezing_samples/Test96Vand_1/']
h5data_location384 = '/data/Freezing_samples/h5data_new/'
h5data_location96 = '/data/Freezing_samples/h5data_new/'
substances384 = ['bact', 'water']
substances96 = ['bact', 'bact', 'water']
wellsize384 = [52, 52]
wellsize96 = [106, 106]
numwells384 = 384
numwells96 = 96
"""
This function goes through the directories. It assumes that 
wells: the wells pictures directory
temperature.csv has the temperatures in accepted format (\t separated, decimal dot)
freezing_points.mat has a humanly distinguished freezing points (ground truth)

stupid_algo_mat.mat has the results of matlab algorithms
"""
import h5py
import scipy.io as spio
import numpy as np
#import csv
import pandas as pd
def load_raw_matlab_data_improved(dirlist, h5data_location, wellsize, numwells, substances, freezing_length=11):
    # freezing_length: an additional parameter that denotes number of timepoint that freezing occures in
     for i in range(len(dirlist)):
         freezing_points=spio.loadmat(dirlist[i]+'freezing_points.mat')
         freeze_nums = freezing_points['freeze_num']
         temps=pd.read_csv(dirlist[i]+'temperature.csv',header=None, delimiter='\t')

#         with open(data_384_dirlist[i]+'temperature.csv') as csv_file:
#             csv_read = csv.reader(csv_file, delimiter='\t')
#             for row in csv_read:
#                   temps = row[:]
         temperatures=[np.asarray(np.float32(temps))]*numwells
         numpoints = temperatures[0].shape[1]
         #temperatures = np.reshape(temperatures, (1, len(temps)))
         well_rows = freeze_nums.shape[0]
         well_columns = freeze_nums.shape[1]
         stupid_algo_results = spio.loadmat(dirlist[i]+'stupid_algo_mat.mat')
         # since h5py now has this fantastic option of virtual datasets what we
         # are going to do is to create a dataset for each sample and then unite
         # them in a virtual dataset
         filename=h5data_location+str(i)+'_raw_dataset_'+str(numwells)+substances[i]+str(i)+'freez'+str(freezing_length)+'.hdf5'
         with h5py.File(filename, 'w') as f:
            print('Saving into:'+filename)
            g1 = f.create_group('Raw_data')
#            wells_list =[{'images':np.empty((wellsize[0],wellsize[1],len(temps)), np.uint8),'temps': temperatures, 
#                           'position': np.empty((2,len(temps)),np.uint8), 'labels': np.empty((1,len(temps)), np.uint8), 
#                           'features': np.empty((14, len(temps)), np.float32), 
#                           'matlab_classif': np.empty((1,len(temps)),np.uint8), 'substance':'', 'comments':''}]*numwells
            images_dtst = [np.empty((wellsize[0],wellsize[1],numpoints), np.uint8)]*numwells
            positions_dtst = np.empty((2,numwells),np.uint8)
            labels_dtst = np.empty((1,numpoints,numwells), np.uint8)
            features_dtst=np.empty((numpoints,14,numwells), np.float32)
            matlab_dtst = np.empty((1,numpoints,numwells),np.uint8)
            substance_dtst = np.empty((1,numwells),np.uint8)
            exclude_dtst = np.zeros((1,numwells),np.bool)
            comments_dtst = ['']*numwells
            d_images = g1.create_dataset('images_dataset', compression="lzf", data=images_dtst)
            d_position = g1.create_dataset('positions_dataset', compression="lzf", data=positions_dtst)
            g1.create_dataset('temperatures_dataset', compression="lzf", data=temperatures)
            d_labels = g1.create_dataset('labels_dataset', compression="lzf", data=labels_dtst)
            d_features = g1.create_dataset('features_dataset', compression="lzf", data=features_dtst)
            d_matlab = g1.create_dataset('matlab_dataset', compression="lzf", data=matlab_dtst)
            d_substance = g1.create_dataset('substance_dataset', compression="lzf", data=substance_dtst)
            d_exclude = g1.create_dataset('exclude_dataset', compression="lzf", data=exclude_dtst)
            g1.attrs['Comments']=comments_dtst
            iterator = 0
            for j in range(well_rows):
                for k in range(well_columns):
                    print(f'Well number {iterator}:j={j},k={k}'.format(iterator,j,k))
                    d_substance[:,iterator] = encode_substances(substances[i])
                    if freeze_nums[j,k] == 0:
                        d_exclude[:,iterator] = 1
                        print(f'->Illegal (0) freezing position: Well excluded')
                    elif(freezing_points["freeze_num"][j,k]-1>numpoints-1):
                        d_exclude[:,iterator] = 1
                        print(f'->Illegal (Too high) freezing position: Well excluded')
                    else:
                        well_read= spio.loadmat(dirlist[i]+\
                                                 '/wells/well_'+str(j+1)+'_'+str(k+1)+'.mat')
                        d_images[iterator] = well_read["well_stack"]
                        d_position[:,iterator] = [j+1,k+1]
                        labels = np.zeros((1,numpoints), np.uint8)
                        fp = min(freezing_points["freeze_num"][j,k]-1, numpoints-1)
                        st_f = max(freezing_points["freeze_num"][j,k]-freezing_length, 0)
                        labels[0,fp] = 1
                        labels[0,st_f:fp] = 2
                        

                        labels[0,fp+1:] = 3 # frozen
                        
                        d_labels[0,:,iterator] = labels
                        features = well_read["features_vec"]
                        d_features[:,:,iterator] = features
                        d_matlab[0,:,iterator] = np.zeros((1,numpoints))
                        st_alg_fr_point = min(stupid_algo_results["after_thresh"][j,k]-1,numpoints-1)
                        st_alg_start_freezing = max(stupid_algo_results["after_thresh"][j,k]-freezing_length,0)
                        if (stupid_algo_results["after_thresh"][j,k]-freezing_length<0)|(stupid_algo_results["after_thresh"][j,k]-1>numpoints-1):
                            print('Warning: matlab algorithm gives wrong numbers')
                        d_matlab[0,st_alg_fr_point,iterator] = 1
                        d_matlab[0,st_alg_start_freezing:st_alg_fr_point,iterator] = 2
                    
                    iterator+=1
def load_chunked_matlab_data_improved(dirlist, h5data_location, wellsize, numwells, substances, chunklength, freezing_length=11):
    # new length is length- (chunklength-1)
    for i in range(len(dirlist)):
         freezing_points=spio.loadmat(dirlist[i]+'freezing_points.mat')
         freeze_nums = freezing_points['freeze_num']
         temps=pd.read_csv(dirlist[i]+'temperature.csv',header=None, delimiter='\t')

#         with open(data_384_dirlist[i]+'temperature.csv') as csv_file:
#             csv_read = csv.reader(csv_file, delimiter='\t')
#             for row in csv_read:
#                   temps = row[:]
         temperatures=[np.asarray(np.float32(temps))]*numwells
         numpoints = temperatures[0].shape[1]
         numpoints_new = numpoints -(chunklength-1)
         new_temp1 = np.zeros((1,numpoints_new))
         for ii in range(numpoints_new):
             new_temp1[0,ii] = np.mean(temperatures[0][:,ii:chunklength])
         new_temperatures = [new_temp1]*numwells
         #temperatures = np.reshape(temperatures, (1, len(temps)))
         well_rows = freeze_nums.shape[0]
         well_columns = freeze_nums.shape[1]
         stupid_algo_results = spio.loadmat(dirlist[i]+'stupid_algo_mat.mat')
         # since h5py now has this fantastic option of virtual datasets what we
         # are going to do is to create a dataset for each sample and then unite
         # them in a virtual dataset
         filename=h5data_location+str(i)+'_chunked_dataset_'+str(numwells)+substances[i]+str(i)+'freez'+str(freezing_length)+'.hdf5'
         with h5py.File(filename, 'w') as f:
            print('Saving into:'+filename)
            g1 = f.create_group('Raw_data')
#            wells_list =[{'images':np.empty((wellsize[0],wellsize[1],len(temps)), np.uint8),'temps': temperatures, 
#                           'position': np.empty((2,len(temps)),np.uint8), 'labels': np.empty((1,len(temps)), np.uint8), 
#                           'features': np.empty((14, len(temps)), np.float32), 
#                           'matlab_classif': np.empty((1,len(temps)),np.uint8), 'substance':'', 'comments':''}]*numwells
            images_dtst = [np.empty((wellsize[0],wellsize[1],chunklength,numpoints_new), np.uint8)]*numwells
            positions_dtst = np.empty((2,numwells),np.uint8)
            labels_dtst = np.empty((1,numpoints_new,numwells), np.uint8)
            features_dtst=np.empty((numpoints_new,14, chunklength,numwells), np.float32)
            matlab_dtst = np.empty((1,numpoints_new,numwells),np.uint8)
            substance_dtst = np.empty((1,numwells),np.uint8)
            exclude_dtst = np.zeros((1,numwells),np.bool)
            comments_dtst = ['']*numwells
            d_images = g1.create_dataset('images_dataset', compression="lzf", data=images_dtst)
            d_position = g1.create_dataset('positions_dataset', compression="lzf", data=positions_dtst)
            g1.create_dataset('temperatures_dataset', compression="lzf", data=new_temperatures)
            d_labels = g1.create_dataset('labels_dataset', compression="lzf", data=labels_dtst)
            d_features = g1.create_dataset('features_dataset', compression="lzf", data=features_dtst)
            d_matlab = g1.create_dataset('matlab_dataset', compression="lzf", data=matlab_dtst)
            d_substance = g1.create_dataset('substance_dataset', compression="lzf", data=substance_dtst)
            d_exclude = g1.create_dataset('exclude_dataset', compression="lzf", data=exclude_dtst)
            g1.attrs['Comments']=comments_dtst
            iterator = 0
            for j in range(well_rows):
                for k in range(well_columns):
                    print(f'Well number {iterator}:j={j},k={k}'.format(iterator,j,k))
                    d_substance[:,iterator] = encode_substances(substances[i])
                    if freeze_nums[j,k] == 0:
                        d_exclude[:,iterator] = 1
                        print(f'->Illegal (0) freezing position: Well excluded')
                    elif(freezing_points["freeze_num"][j,k]-1>numpoints-1):
                        d_exclude[:,iterator] = 1
                        print(f'->Illegal (Too high) freezing position: Well excluded')
                    else:
                        well_read= spio.loadmat(dirlist[i]+\
                                                 '/wells/well_'+str(j+1)+'_'+str(k+1)+'.mat')
                        well_temp = well_read["well_stack"]
                        well_chunked = np.empty((wellsize[0],wellsize[1],chunklength,numpoints_new), np.uint8)
                        for ii in range(numpoints_new):
                            well_chunked[:,:,:,ii] = well_temp[:,:,ii:ii+chunklength]
                        d_images[iterator] = well_chunked
                        d_position[:,iterator] = [j+1,k+1]
                        labels = np.zeros((1,numpoints_new), np.uint8)
                        
                        fp = min(freezing_points["freeze_num"][j,k]-1, numpoints-1)
                        st_f = max(freezing_points["freeze_num"][j,k]-freezing_length, 0)
                        new_fp = fp-chunklength
                        new_st_f = st_f-chunklength
                        # 0 - is not frozen
                        labels[0,new_fp] = 1 #freezing point
                        labels[0,new_st_f:new_fp] = 2 # in process of freezing
                        labels[0,new_fp+1:] = 3 # frozen
                        d_labels[0,:,iterator] = labels
                        features_temp = well_read["features_vec"]
                        features_chunked = np.empty((numpoints_new,14, chunklength), np.float32)
                        for ii in range(numpoints_new):
                            temp =features_temp[ii:ii+chunklength,:]
                            features_chunked[ii,:,:] = temp.T # check the axis
                        d_features[:,:,:,iterator] = features_chunked
                        d_matlab[0,:,iterator] = np.zeros((1,numpoints_new))
                        st_alg_fr_point = min(stupid_algo_results["after_thresh"][j,k]-1,numpoints-1)
                        st_alg_start_freezing = max(stupid_algo_results["after_thresh"][j,k]-freezing_length,0)
                        new_fp_st_alg = st_alg_fr_point-chunklength #<--
                        new_st_f_st_alg = st_alg_start_freezing-chunklength
                        
                        if (stupid_algo_results["after_thresh"][j,k]-freezing_length<0)|(stupid_algo_results["after_thresh"][j,k]-1>numpoints-1):
                            print('Warning: matlab algorithm gives wrong numbers')
                        d_matlab[0,new_fp_st_alg,iterator] = 1
                        d_matlab[0,new_st_f_st_alg:new_fp_st_alg,iterator] = 2
                        d_matlab[0,new_fp_st_alg+1:,iterator] = 3
                    
                    iterator+=1
    
            
                 
def encode_substances(substance):
    return {
        'bact': 1,
        'water': 0
        }.get(substance, 0)
    
# Start from 384
load_raw_matlab_data_improved(data_384_dirlist, h5data_location384, wellsize384, numwells384, substances384, 31)
load_raw_matlab_data_improved(data_96_dirlist, h5data_location96, wellsize96, numwells96, substances96, 31)
#load_chunked_matlab_data_improved(data_384_dirlist, h5data_location384, wellsize384, numwells384, substances384, 6, 31)
#load_chunked_matlab_data_improved(data_96_dirlist, h5data_location96, wellsize96, numwells96, substances96, 6, 31)

