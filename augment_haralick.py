#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 20:34:41 2019

@author: andrey
"""
import h5py
import numpy as np
import imutils
from scipy import ndimage
import cv2
import platform
import mahotas as mt
import random
#from matplotlib import pyplot as plt
if platform.node()=='choo-desktop':
    from branch_init_choo import datadir
elif platform.node()=='andrey-cfin':
    from branch_init_cfin import datadir
# just for example
ang_num =2
angles = np.arange(0,360,360/ang_num)
dataset_to_read = datadir+'0_raw_dataset_384bact0freez31.hdf5'
dataset_to_write = datadir+'0_raw_dataset_384bact0freez31_aug.hdf5'
with h5py.File(dataset_to_read, "r", libver="latest") as f1, h5py.File(dataset_to_write, "w", libver="latest") as f2:
    images_d = f1['Raw_data/images_dataset']
    labels_d = f1['Raw_data/labels_dataset']
    features_d = f1['Raw_data/features_dataset']
    exclude_d = f1['Raw_data/exclude_dataset']
    
    print(images_d.shape)
    print(features_d.shape)
    num_wells_orig = images_d.shape[0]
    labels_aug = np.empty((1,labels_d.shape[1], angles.shape[0]*labels_d.shape[2]))
    features_aug = np.empty((labels_d.shape[1],13,num_wells_orig*angles.shape[0]))
    images_aug = np.empty((num_wells_orig*angles.shape[0],images_d.shape[1],images_d.shape[2],labels_d.shape[1]))
    aug_well = 0
    for angle in angles:
        # for each well in original database we will augment
        fps = np.argmax(labels_d[0,:,:], axis=0)
        labels = labels_d[0,:,:]
        
        for i in range(2):#range(num_wells_orig):
            imstack = images_d[i,:,:,:]
            length_im = imstack.shape[2]
            imstack_rotated = imstack.copy()
            zeros_im = imstack[:,:,0]
            gray = cv2.GaussianBlur(zeros_im, (3, 3), 0)
            edged = cv2.Canny(gray, 20, 100)
            im_shape = images_d.shape
            
            # find contours in the edge map
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
    
            # ensure at least one contour was found
            if len(cnts) > 0:
                #grab the largest contour, then draw a mask for the well
                c = max(cnts, key=cv2.contourArea)
                mask = np.zeros(gray.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                 
                 #compute its bounding box of well, then extract the ROI,
                 #and apply the mask
                (x, y, w, h) = cv2.boundingRect(c)

            else:
                print("No contours found! Cannot rotate!")
            hor = 20
            vert = np.ceil(imstack.shape[2]/20)
            for timepoint in range(imstack.shape[2]):
                image = imstack[:,:,timepoint]
                feature_gt = features_d[timepoint,:,i]
                imageROI = image[y:y + h, x:x + w]
                image_rotated = image.copy()
                rotated = ndimage.rotate(imageROI , angle, reshape=False, cval=255)
                image_rotated[y:y + h, x:x + w] = rotated
                mean = 0
                var = 0.05
                sigma = var**0.5
                gauss = np.random.normal(mean,sigma,image_rotated.shape)
                noisy_rotated = np.uint8(image_rotated + gauss)
                imstack_rotated[:,:,timepoint] = noisy_rotated
                #plt.subplot(hor, vert, timepoint+1), plt.imshow(noisy_rotated)
                # cv2.imshow(str(timepoint), noisy_rotated)
                # cv2.waitKey(50)
                # cv2.destroyAllWindows()
            # now let us move the freezing point for the augmented data
            # we will generate a random number of steps +/- rand_width
            # that the freezing point will be moved to if it is positive
            # we will multiply the first frame, plus gaussian noise
            # if negative we will multiply the last one
            # we will update the labels
            rand_width = 20
            rand_shift=np.int8(random.uniform(-rand_width, rand_width))
            while fps[i]+rand_shift<0 and fps[i]+rand_shift>length_im:
                rand_shift=np.int8(random.uniform(-rand_width, rand_width))
            
            imstack_shifted = np.uint8(np.empty((imstack_rotated.shape[0], imstack_rotated.shape[1],
                                        imstack_rotated.shape[2]+abs(rand_shift))))
            gauss = np.uint8(np.random.normal(mean,sigma,(imstack_rotated.shape[0], imstack_rotated.shape[1],
                                        abs(rand_shift))))
            labels_shifted = np.uint8(np.empty((1,labels.shape[0]+abs(rand_shift))))
            if rand_shift >0:
                imstack_shifted[:,:,:rand_shift] =np.tile(imstack_rotated[:,:,0].reshape(
                    imstack_rotated.shape[0],imstack_rotated.shape[1],1), (1,1,rand_shift))+gauss
                imstack_shifted[:,:,rand_shift:]=imstack_rotated
                labels_shifted[:,:rand_shift]=0
                labels_shifted[:,rand_shift:]=labels[:,i]
            elif rand_shift <0:
                imstack_shifted[:,:,rand_shift:] =np.tile(imstack_rotated[:,:,-1].reshape(
                    imstack_rotated.shape[0],imstack_rotated.shape[1],1), (1,1,-rand_shift))+gauss
                imstack_shifted[:,:,:rand_shift]=imstack_rotated
                labels_shifted[:,rand_shift:]=labels[0,-1]
                labels_shifted[:,:rand_shift]=labels[:,i]
            
            labels_new=labels_shifted[:,:labels.shape[0]]

            imstack_new =imstack_shifted[:,:,:labels.shape[0]]
            
            # now let us calculate the haralick parameters
            images_aug[aug_well,:,:,:]=imstack_new
            labels_aug[0,:,aug_well]=labels_new
            # and push them to the database
            
            for j in range(imstack_new.shape[2]):
                None
                #features_aug[j,:,aug_well]=np.mean(mt.features.haralick(image), axis=0)
            #let us check the pictures (movie?)
            aug_well+=1
    f2.create_dataset("Raw_data/features2_dataset", compression=8, data=features_aug)
    f2.create_dataset("Raw_data/images_dataset", compression=8, data=images_aug)
    f2.create_dataset("Raw_data/labels_dataset", compression=8, data=labels_aug)
    

    None
