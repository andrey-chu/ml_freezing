#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 19:50:13 2019

@author: andrey
"""

import h5py
import numpy as np
import imutils
from scipy import ndimage
import cv2
hd5py_dir = "/media/backup/data/Freezing_samples/h5data_new/"
# just for example
dataset_to_read = hd5py_dir+'0_raw_dataset_384bact0freez31.hdf5'
#dataset_to_write = hd5py_dir+'0_raw_dataset_384bact0freez31_aug.hdf5'
ang_num =70
angles = np.arange(0,360,360/ang_num) 
with h5py.File(dataset_to_read, "r", libver="latest") as f1:
    images_d = f1['Raw_data/images_dataset']
    #    position_d = f1['Raw_data/positions_dataset']
    #    temperatures_d =f1['Raw_data/temperatures_dataset']
    labels_d = f1['Raw_data/labels_dataset']
    #    features_d = f1['Raw_data/features_dataset']
    #    matlab_d = f1['Raw_data/matlab_dataset']
    #    substance_d = f1['Raw_data/substance_dataset']
    #    exclude_d = f1['Raw_data/exclude_dataset']
    #    "Raw_data/datasets_dataset"
    imstack = images_d[0,:,:,:]
    print(images_d.shape)
    image = imstack[:,:,1]
    gray = cv2.GaussianBlur(image, (3, 3), 0)
    edged = cv2.Canny(gray, 20, 100)
    im_shape = images_d.shape
    
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # ensure at least one contour was found
    if len(cnts) > 0:
    	# grab the largest contour, then draw a mask for the well
    	c = max(cnts, key=cv2.contourArea)
    	mask = np.zeros(gray.shape, dtype="uint8")
    	cv2.drawContours(mask, [c], -1, 255, -1)
     
    	# compute its bounding box of well, then extract the ROI,
    	# and apply the mask
    	(x, y, w, h) = cv2.boundingRect(c)
        # rect_mask = np.zeros(image.shape)
        # rect_mask[y:y + h, x:x + w]=1
    	imageROI = image[y:y + h, x:x + w]
    	# maskROI = mask[y:y + h, x:x + w]
    	# imageROI1= cv2.bitwise_and(imageROI, imageROI, mask=maskROI)
        
    image_rotated = image.copy()
        
    for angle in angles:
            rotated = ndimage.rotate(imageROI , angle, reshape=False, cval=255)
            image_rotated[y:y + h, x:x + w] = rotated
            
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,image_rotated.shape)
          
            noisy = np.uint8(image_rotated + gauss)
            cv2.imshow("Rotated", noisy)
            cv2.waitKey(0)
# rotated = imutils.rotate(imageROI , angles[10])
# cv2.imshow("Mask", maskROI)
# cv2.imshow("Mask1", rect_mask)
cv2.destroyAllWindows()