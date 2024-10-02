# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:01:04 2023

@author: emahu
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import morphology
import czifile


def diffuse_backgr_filter_2d(img2d, sigma=0.5):
    """Take 2D image and perform diffuse background removal, return binary mask (Otsu threshol)"""

    # median filter with different radius of used footprint (circle)
    sec_med =[]
    
    for radius in range(1,10,2):
        
        circle = morphology.disk(radius)
    
        sec_med.append(filters.median(img2d, footprint=circle))
    
    # add original section to the stack
    sec_med.append(img2d) 
    sec_med_stack = np.stack((sec_med))   
     
    # choose min for each pixel in the stack to the resulting img
    min_med = np.min(sec_med_stack, axis=0)
        
    # diffuse background removal
    img_diffused = img2d - min_med 
    
    # # remove high frequency noise
    # img_gaus = filters.gaussian(img_diffused, sigma)
    
    # # use Otsu threshold and create binary mask
    # thr_otsu = filters.threshold_otsu(img_gaus)
    # thr_mask = img_gaus >= thr_otsu  
    
    return img_diffused

def mit_morpho_2d(thr_mask, radius = 1, remove_small_obj = 50):
    
    if radius == None:
        img_closed = morphology.binary_closing(morphology.binary_dilation(thr_mask))
    else:   
        disk = morphology.disk(radius)
        img_closed = morphology.binary_closing(morphology.binary_dilation(thr_mask), footprint = disk)
    
    img_remove_small = morphology.remove_small_objects(img_closed, min_size = remove_small_obj)
    
    return img_remove_small






