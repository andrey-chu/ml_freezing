#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:48:04 2019

@author: andrey
"""
import h5py
import numpy as np
from svm_freezing_seglearn import extract_haralick
import platform
if platform.node()=='choo-desktop':
    from branch_init_choo import datadir
elif platform.node()=='andrey-cfin':
    from branch_init_cfin import datadir

def unite_datasets(list_to_unite, united_name, raw_chunked):
    if raw_chunked == 'raw':
        shape_tmp = np.zeros((len(list_to_unite), 4))
    elif raw_chunked == 'chunked':
        shape_tmp = np.zeros((len(list_to_unite), 5))
    
    for i in range(len(list_to_unite)):
        # there are folowing datasets:
        # - 
        # first parse through list and collect sizes
        # then concatenate
        # the major sizes we will save in our dataset:
        # wellsize[0], wellsize[1]
        # numpoints
        # numwells

        with h5py.File(list_to_unite[i], 'r') as f:
            images_d = f['Raw_data/images_dataset']
            position_d = f['Raw_data/positions_dataset']
            temperatures_d =f['Raw_data/temperatures_dataset']
            labels_d = f['Raw_data/labels_dataset']
            features_d = f['Raw_data/features_dataset']
            matlab_d = f['Raw_data/matlab_dataset']
            substance_d = f['Raw_data/substance_dataset']
            exclude_d = f['Raw_data/exclude_dataset']
            shape_tmp[i] = images_d.shape
#            print(substance_d.shape)
#            print(labels_d.dtype)
    #        t_shape[i]=temperatures_d.shape
            
            
    united_shape = np.max((shape_tmp), axis=0)
    if raw_chunked == 'chunked':
        final_shape_images = (int(np.sum(shape_tmp[:,0])),int(united_shape[1]),int(united_shape[2]),int(united_shape[3]),int(united_shape[4]))
    elif raw_chunked == 'raw':
        final_shape_images = (int(np.sum(shape_tmp[:,0])),int(united_shape[1]),int(united_shape[2]),int(united_shape[3]))
    maxdatalines=int(united_shape[-1])
    totalwells=final_shape_images[0]
    if raw_chunked == 'chunked':
        chunklength = final_shape_images[3]
    
    
    with h5py.File(united_name, "rw", libver="latest") as f1:
        g1 = f1.create_group('Raw_data')
        shapes_size = [int(np.sum(shape_tmp[:,0])), shape_tmp.shape[1]-2]
        shapes_placeholder = np.ones((shapes_size))
        d_shapes = g1.create_dataset('shapes_dataset', data=shapes_placeholder)
        temp = 0
        for iter1 in range(shape_tmp.shape[0]):
            d_shapes[temp:int(shape_tmp[iter1,0]+temp), 0]=shape_tmp[iter1,1]
            d_shapes[temp:int(shape_tmp[iter1,0]+temp), 1]=shape_tmp[iter1,3]
            if raw_chunked == 'chunked':
                d_shapes[temp:int(shape_tmp[iter1,0]+temp), 2]=shape_tmp[iter1,4]
            temp+=int(shape_tmp[iter1,0])
        layout_images = h5py.VirtualLayout(shape=final_shape_images, dtype="u1")
        layout_position=h5py.VirtualLayout(shape=(2,final_shape_images[0]), dtype="u1")
        layout_temperatures=h5py.VirtualLayout(shape=(int(united_shape[-1]),int(final_shape_images[0])), dtype="f4")
        layout_labels=h5py.VirtualLayout(shape=(maxdatalines,totalwells), dtype="u1")
        if raw_chunked == 'chunked':
            layout_features=h5py.VirtualLayout(shape=(maxdatalines, 14, chunklength, totalwells), dtype="f4")
        else:
            layout_features=h5py.VirtualLayout(shape=(maxdatalines, 14, totalwells), dtype="f4")
        layout_matlab=h5py.VirtualLayout(shape=(maxdatalines, totalwells), dtype="u1")
        layout_substance=h5py.VirtualLayout(shape=(1,totalwells), dtype="u1")
        layout_exclude=h5py.VirtualLayout(shape=(1,totalwells), dtype=np.bool)
        iter1 = 0
        datasets_dtst = np.zeros((0,))
        for i in range(len(list_to_unite)):
            currentwells=int(shape_tmp[i, 0])
            currentpoints=int(shape_tmp[i, -1])
            vsource_images = h5py.VirtualSource(list_to_unite[i], "Raw_data/images_dataset", shape=shape_tmp[i])
            
            if raw_chunked == 'chunked':
                layout_images[int(iter1):int(shape_tmp[i, 0]+iter1),:,:,:,:int(shape_tmp[i, -1])]=vsource_images
                vsource_features= h5py.VirtualSource(list_to_unite[i], "Raw_data/features_dataset",\
                                                  shape=(currentpoints, 14, chunklength, currentwells))
                layout_features[:currentpoints,:,:,int(iter1):int(currentwells+iter1)] = vsource_features
            elif raw_chunked == 'raw':
                layout_images[int(iter1):int(shape_tmp[i, 0]+iter1),:,:,:int(shape_tmp[i, -1])]=vsource_images
                vsource_features= h5py.VirtualSource(list_to_unite[i], "Raw_data/features_dataset",\
                                                  shape=(currentpoints, 14,  currentwells))
                layout_features[:currentpoints,:,int(iter1):int(currentwells+iter1)] = vsource_features
                
           # layout_images[i]=vsource_images
            vsource_position = h5py.VirtualSource(list_to_unite[i], "Raw_data/position_dataset",\
                                                  shape=(2,int(shape_tmp[i, 0])))
            layout_position[:,int(iter1):int(currentwells+iter1)]=vsource_position
            vsource_temperatures = h5py.VirtualSource(list_to_unite[i], "Raw_data/temperatures_dataset",\
                                                  shape=(int(shape_tmp[i, 0]),1,int(shape_tmp[i, -1])))
            layout_temperatures[:int(shape_tmp[i, -1]),int(iter1):int(shape_tmp[i, 0]+iter1)] = vsource_temperatures
            vsource_labels = h5py.VirtualSource(list_to_unite[i], "Raw_data/labels_dataset",\
                                                  shape=(1, currentpoints,currentwells))
            layout_labels[:currentpoints,int(iter1):int(currentwells+iter1)] = vsource_labels
            
            
            vsource_matlab = h5py.VirtualSource(list_to_unite[i], "Raw_data/matlab_dataset",\
                                                  shape=(1, currentpoints, currentwells))
            layout_matlab[:currentpoints,int(iter1):int(currentwells+iter1)] = vsource_matlab
            vsource_substance = h5py.VirtualSource(list_to_unite[i], "Raw_data/substance_dataset",\
                                                  shape=(1, currentwells))
            layout_substance[0,int(iter1):int(currentwells+iter1)] = vsource_substance
            vsource_exclude = h5py.VirtualSource(list_to_unite[i], "Raw_data/exclude_dataset",\
                                                  shape=(1, currentwells))
            layout_exclude[0,int(iter1):int(currentwells+iter1)] = vsource_exclude
            iter1 +=shape_tmp[i, 0]
            add_to_datasets = np.ones((int(shape_tmp[i, 0]),))*i
            datasets_dtst = np.concatenate((datasets_dtst, add_to_datasets))
        f1.create_virtual_dataset("Raw_data/images_dataset", layout_images, fillvalue=-5)
        f1.create_virtual_dataset("Raw_data/position_dataset", layout_position, fillvalue=-5)
        #next line is problematic check fillvalue
        f1.create_virtual_dataset("Raw_data/temperatures_dataset", layout_temperatures, fillvalue=np.nan)
        f1.create_virtual_dataset("Raw_data/labels_dataset", layout_labels, fillvalue=100)
        f1.create_virtual_dataset("Raw_data/features_dataset", layout_features, fillvalue=np.nan)
        f1.create_virtual_dataset("Raw_data/matlab_dataset", layout_matlab, fillvalue=100)
        f1.create_virtual_dataset("Raw_data/substance_dataset", layout_substance, fillvalue=-5)
        f1.create_virtual_dataset("Raw_data/exclude_dataset", layout_exclude, fillvalue=-5)
        f1.create_dataset("Raw_data/datasets_dataset", compression="lzf", data=datasets_dtst)
        img = f1["Raw_data/images_dataset"]
        features_2 = extract_haralick(img)
        f1.create_dataset("Raw_data/features2_dataset", compression="lzf", data=features_2)
hd5py_dir = datadir
#unite_datasets([hd5py_dir+'0_chunked_dataset_384bact0freez31.hdf5', hd5py_dir+'1_chunked_dataset_384water1freez31.hdf5'], hd5py_dir+'united_chunked_dataset_384freez31.hdf5', 'chunked')
#unite_datasets([hd5py_dir+'0_chunked_dataset_96bact0freez31.hdf5', hd5py_dir+'1_chunked_dataset_96bact1freez31.hdf5', hd5py_dir+'2_chunked_dataset_96water2freez31.hdf5'], hd5py_dir+'united_chunked_dataset_96freez31.hdf5', 'chunked')
unite_datasets([hd5py_dir+'0_raw_dataset_384bact0freez31.hdf5', hd5py_dir+'1_raw_dataset_384water1freez31.hdf5'], hd5py_dir+'united_raw_dataset_384freez31f2.hdf5', 'raw')
unite_datasets([hd5py_dir+'0_raw_dataset_96bact0freez31.hdf5', hd5py_dir+'1_raw_dataset_96bact1freez31.hdf5', hd5py_dir+'2_raw_dataset_96water2freez31.hdf5'], hd5py_dir+'united_raw_dataset_96freez31f2.hdf5', 'raw')
#unite_datasets(['/data/Freezing_samples/h5data/0_chunked_dataset_384bact0.hdf5','/data/Freezing_samples/h5data/1_chunked_dataset_384water1.hdf5'], '/data/Freezing_samples/h5data/united_dat_384.hdf5', 'chunked')
#unite_datasets(['/data/Freezing_samples/h5data/0_raw_dataset_384bact0.hdf5','/data/Freezing_samples/h5data/1_raw_dataset_384water1.hdf5'], '/data/Freezing_samples/h5data/united_dat_384_test-raw.hdf5', 'raw')
#unite_datasets(['/data/Freezing_samples/h5data/0_chunked_dataset_384bact0.hdf5','/data/Freezing_samples/h5data/0_chunked_dataset_384bact0.hdf5'], '/data/Freezing_samples/h5data/united_dat.hdf5', 'chunked')

# it would be better to rearrange the chunky version (or at least the 'heavy' databases of images and features)
#into the form of all_chunks x [features] and [all_chunks x images] in this way we can read it gradually more easily from the database