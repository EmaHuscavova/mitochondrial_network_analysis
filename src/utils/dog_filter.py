# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:18:19 2023

@author: emahu
"""

from skimage.util import img_as_float
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import peak_local_max
from scipy.signal import find_peaks 
import numpy as np




def DoG(image, min_sigma, max_sigma, sigma_ratio=1.6):
    
    image = img_as_float(image)
    
    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])
    
    print(sigma_list)
    
    # computing difference between two successive Gaussian blurred images
    # to obtain an approximation of the scale invariant Laplacian of the
    # Gaussian operator
    dog_image_cube = np.empty(image.shape + (k,))
    gaussian_previous = gaussian(image, sigma_list[0], mode='reflect')
    for i, s in enumerate(sigma_list[1:]):
        gaussian_current = gaussian(image, s, mode='reflect')
        dog_image_cube[..., i] = gaussian_previous - gaussian_current
        gaussian_previous = gaussian_current
        
    # normalization factor for consistency in DoG magnitude
    sf = 1 / (sigma_ratio - 1)
    dog_image_cube *= sf
    
    
    res = np.max(dog_image_cube, axis=2)
    plt.figure(figsize=(10,10))
    plt.imshow(res)
    return res



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import tifffile as tf

    img = tf.imread(r"..\test\CTRL_05_C001.tif")
    img = img[:,:,0]
    plt.figure(figsize=(10,10))
    plt.imshow(img)

    min_sigma = 1
    max_sigma = 7.5
    sigma_ratio=1.6

    res = DoG(img, min_sigma, max_sigma, sigma_ratio=1.6)

    thr_otsu = threshold_otsu(res)

    thr_mask = res >= 0.08
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.imshow(thr_mask, alpha= 0.5)
        
        
        
    
        

