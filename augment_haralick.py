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
#import mahotas as mt
import random
#from matplotlib import pyplot as plt
if platform.node()=='choo-desktop':
    from branch_init_choo import datadir
elif platform.node()=='andrey-cfin':
    from branch_init_cfin import datadir
from supp_methods import extract_haralick, extract_haralick_parallel
augmented_video_dir = datadir+"../aug/"
#np.random.seed(10)
#random.seed(10)
# just for example
ang_num =10

start_ang =0#2
angles = range(ang_num)#np.arange(start_ang,360,(360-start_ang)/ang_num)
dataset_to_read = datadir+'0_raw_dataset_384bact0freez31.hdf5'
dataset_to_write = datadir+'0_raw_dataset_384bact0freez31_aug1.hdf5'
with h5py.File(dataset_to_read, "r", libver="latest") as f1, h5py.File(dataset_to_write, "w", libver="latest") as f2:
    images_d = f1['Raw_data/images_dataset']
    labels_d = f1['Raw_data/labels_dataset']
    features_d = f1['Raw_data/features_dataset']
    exclude_d = f1['Raw_data/exclude_dataset']
    temps_d = f1['Raw_data/temperatures_dataset']
    substance_d= f1['Raw_data/substance_dataset']
    temps = temps_d[:]
    excluded = exclude_d[:]
     
    print(exclude_d.shape)
    #print(features_d.shape)
    # import pdb; pdb.set_trace()
    num_wells_orig = images_d.shape[0]
    rand_angles=start_ang+np.random.rand(ang_num,num_wells_orig)*(360-start_ang)
    labels_aug = np.empty((1,labels_d.shape[1], ang_num*labels_d.shape[2]))
    features_aug = np.empty((labels_d.shape[1],13,num_wells_orig*ang_num))
    images_aug = np.empty((num_wells_orig*ang_num,images_d.shape[1],images_d.shape[2],labels_d.shape[1]), dtype=np.uint8)
    excluded_aug = np.empty((1,num_wells_orig*ang_num))
    positions_aug = np.empty((2,num_wells_orig*ang_num))
    temps_aug = np.empty((num_wells_orig*ang_num, 1, labels_d.shape[1]))
    substance_aug = np.empty((1,num_wells_orig*ang_num))
    aug_well = 0

    for angle in angles:
        # for each well in original database we will augment
        fps = np.argmax(labels_d[0,:,:], axis=0)
        labels = labels_d[0,:,:]
        
        for i in range(num_wells_orig):

            temps_aug[aug_well, 0, :] = temps[i, 0, :]
            substance_aug[0,aug_well]=101
            excluded_aug[0,aug_well] = excluded[0,i]
            positions_aug[:,aug_well] = [0,0]
            imstack = images_d[i,:,:,:]
            length_im = imstack.shape[2]
            imstack_rotated = imstack.copy()
            imstack_rotated1 = imstack.copy()
            zeros_im = imstack[:,:,0]
            gray = cv2.GaussianBlur(255-zeros_im, (3, 3), 0)
            edged = cv2.Canny(gray, 20, 100)
            im_shape = images_d.shape
            
            # Save the video of original well
            # w,h,l=imstack.shape
            # video_n = str(i)+"orig.avi"
            # video_name = augmented_video_dir+video_n
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # video=cv2.VideoWriter(video_name, fourcc, 40, (w,h),isColor=False)
            # for k in range(l):
            #     video.write(imstack[:,:,k])
            # video.release()
            # cv2.destroyAllWindows()
            
            
            
            
            # find contours in the edge map
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            cnts1 = imutils.grab_contours(cnts)
    
            # ensure at least one contour was found
            if len(cnts1) > 0:
                #grab the largest contour, then draw a mask for the well
                c = max(cnts1, key=cv2.contourArea)
                mask = np.zeros(gray.shape, dtype="uint8")
                #cv2.drawContours(mask, [c], -1, 255, -1)
                cv2.drawContours(mask, cnts1, -1, 255, -1)
                 #compute its bounding box of well, then extract the ROI,
                 #and apply the mask
                (x, y, w, h) = cv2.boundingRect(mask)

            else:
                print(str(i)+": No contours found! Cannot rotate! Excluded?"+str(excluded[0,i])+ "->Excluding")
                excluded_aug[0,aug_well]=True
                
            hor = 20
            vert = np.ceil(imstack.shape[2]/20)
            for timepoint in range(imstack.shape[2]):
                image = imstack[:,:,timepoint]
                feature_gt = features_d[timepoint,:,i]
                imageROI = image[y:y + h, x:x + w]
                image_rotated = image.copy()
                cur_angle= rand_angles[angle, i]
                rotated = ndimage.rotate(imageROI, cur_angle, reshape=False, cval=255)
                image_rotated[y:y + h, x:x + w] = rotated
                imstack_rotated1[:,:,timepoint] = image_rotated
            mean = 0
            var = 0.005
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,imstack_rotated1.shape)
            #noisy=imstack_rotated1 + gauss

            imstack_rotated1[imstack_rotated1<0]=0
            imstack_rotated1[imstack_rotated1>254]=255
                
            noisy_rotated = np.uint8(imstack_rotated1)
            # if aug_well==112:#noisy_rotated[noisy_rotated==0].size>0.5*noisy_rotated.size:
            #     import pdb; pdb.set_trace()
            imstack_rotated = noisy_rotated
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
            rand_width = 40
            # import pdb; pdb.set_trace()
            rand_shift=np.int8(random.uniform(-rand_width, rand_width))
            while fps[i]+rand_shift<0 and fps[i]+rand_shift>length_im:
                rand_shift=np.int8(random.uniform(-rand_width, rand_width))
            
            imstack_shifted = np.uint8(np.empty((imstack_rotated.shape[0], imstack_rotated.shape[1],
                                        imstack_rotated.shape[2]+abs(rand_shift))))
            # gauss = np.uint8(np.random.normal(mean,sigma,(imstack_rotated.shape[0], imstack_rotated.shape[1],
            #                             abs(rand_shift))))
            labels_shifted = np.uint8(np.empty((1,labels.shape[0]+abs(rand_shift))))
            if rand_shift >0:
                imstack_shifted[:,:,:rand_shift] =np.tile(imstack_rotated[:,:,0].reshape(
                    imstack_rotated.shape[0],imstack_rotated.shape[1],1), (1,1,rand_shift))
                imstack_shifted[:,:,rand_shift:]=imstack_rotated
                labels_shifted[:,:rand_shift]=0
                labels_shifted[:,rand_shift:]=labels[:,i]
            elif rand_shift <0:
                imstack_shifted[:,:,rand_shift:] =np.tile(imstack_rotated[:,:,-1].reshape(
                    imstack_rotated.shape[0],imstack_rotated.shape[1],1), (1,1,-rand_shift))
                imstack_shifted[:,:,:rand_shift]=imstack_rotated
                labels_shifted[:,rand_shift:]=labels[0,-1]
                labels_shifted[:,:rand_shift]=labels[:,i]
            else:
                labels_shifted = np.reshape(labels[:,i], (1, labels.shape[0]))
                imstack_shifted = imstack_rotated
            
            
            
            labels_new=labels_shifted[:,:labels.shape[0]]

            imstack_new =imstack_shifted[:,:,:labels.shape[0]]
            noisy=imstack_new+gauss
 
            images_aug[aug_well,:,:,:]=noisy
            labels_aug[0,:,aug_well]=labels_new
            
             #let us check the pictures (movie?)
            
            ### save the movies
            w,h,l=imstack_new.shape
            video_n = "well_{0}_ang{1}_{2:.1f}.avi".format(i,angle,cur_angle)
            #video_n = "well_"+str(i)+"ang"+str(angle)+".avi"
            video_name = augmented_video_dir+video_n
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video=cv2.VideoWriter(video_name, fourcc, 40, (w,h),isColor=False)
            #import pdb; pdb.set_trace()
            for k in range(l):
                video.write(imstack_new[:,:,k])
            video.release()
            cv2.destroyAllWindows()
           ### end save the movies
            aug_well+=1
    # now let us calculate the haralick parameters
    features_aug = extract_haralick_parallel(images_aug)
    # and push them to the database
    f2.create_dataset("Raw_data/features2_dataset", compression=8, data=features_aug)
    f2.create_dataset("Raw_data/images_dataset", compression=8, data=images_aug)
    f2.create_dataset("Raw_data/labels_dataset", compression=8, data=labels_aug)
    f2.create_dataset("Raw_data/exclude_dataset", compression=8, data=excluded_aug)
    f2.create_dataset("Raw_data/features_dataset", compression=8, data=features_aug)
    f2.create_dataset("Raw_data/matlab_dataset", compression=8, data=labels_aug)
    f2.create_dataset("Raw_data/position_dataset", compression=8, data=positions_aug)
    f2.create_dataset("Raw_data/substance_dataset", compression=8, data=substance_aug)
    f2.create_dataset("Raw_data/temperatures_dataset", compression=8, data=temps_aug)
    
    None
