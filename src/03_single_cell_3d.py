# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:31:58 2023

@author: emahu
"""


import utils.utils3d as ut3d

import tifffile as tiff 
import numpy as np
import scipy.ndimage as ndim
from skimage.io import imread
import glob
import os


def cut_roi_3d(img, image_name):
    '''
    
    Cut regions of interest (ROIs) from a max projection and save the resulting 
    single cell images as NumPy arrays.

    Parameters:
    - img (numpy.ndarray): NumPy array representing the input max projection 
        image.
    - image_name (str): Name of the input image file.

    Returns:
    - None

    Notes:
    - This function loads ROIs from a specified folder and masks the input 
        image accordingly to isolate cell regions.
    - The resulting cell images are saved as NumPy arrays.

    '''
    # Define the image type (either '.png' or '.tif')
    img_type = '.tif' 
    
    # previously data divided to folders by treatment
    # treatment = '20uM'

    # ROIs_folder = rf"C:\Users\emahu\Documents\mitochondria_project_2023\analysis\Kanako\max_projs\{treatment}\masks"
    
    # data contain info about treatment in the image name
    # Specify the folder containing ROIs
    ROIs_folder = r"..\max_projs\sample\masks"
    
    # Iterate over all ROI files in the ROIs_folder
    for roi_path in glob.glob( os.path.join( ROIs_folder, image_name.replace('.npy', f'*{img_type}') ), recursive=True ): 
        
        if img_type == '.png':
            roi = imread(roi_path)
        elif img_type == '.tif':
            roi = tiff.imread(roi_path)
        
        print(f'roi: {roi_path}')
        
        # Extract the red channel of the ROI (assuming it's a RGB image)
        roi_red = roi[:,:,0]
        
        # Create a binary mask from the red channel
        roi_mask = np.zeros_like(roi_red)
        roi_mask[ roi_red > 0] = 1
        roi_mask = ndim.binary_fill_holes(roi_mask)
        
        # Mask the input image to isolate cell regions
        cell = img * roi_mask
        
        # Specify the path to save the resulting cell image
        # cell_path = roi_path.replace(rf'max_projs\{treatment}\masks', rf'single_cell_imgs\{treatment}')
        cell_path = roi_path.replace( r'max_projs\masks', r'single_cells')
        cell_path = cell_path.replace(f'{img_type}', '.npy')
        
        # Save the resulting cell image as a NumPy array
        print(f'save: {cell_path}')
        np.save(cell_path, cell)
        
        
        
# path to preprocessed 3d imgs ( to obtain relat path inside the np_arr_analysis function )
source_folder= r"..\preprocessed\sample"

dest_folder = r"..\single_cells\sample"

actions = [ ('cut_roi_3d', cut_roi_3d) ] # input in the np_arr_analysis func

ut3d.np_arr_analysis(source_folder, dest_folder, actions, input_img_type = 'mit', save_as = None)

# TODO: save is not working correctly


# # see results
# import napari
# images = ut3d.load_npy(dest_folder, im_type = 'max_proj*', return_type= 'dictionary')

# viewer = napari.Viewer(ndisplay=3)
# for name, img in images.items():
#     viewer.add_image(img, name=f'{name}')
                      


