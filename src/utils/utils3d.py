# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:06:52 2023

@author: emahu
"""

import numpy as np
from skimage import morphology
from skimage import filters
import scipy.ndimage as ndim
import czifile
import glob
import os
from aicsimageio import AICSImage
from aicsimageio import metadata
import tifffile
from skimage.filters import meijering, sato, frangi, hessian, threshold_triangle
from skimage.morphology import skeletonize


def get_pixel_size(image_path):
    '''Return physical size of pixel from metadata, to determine zoom_coefs'''
    img_test = AICSImage(image_path)
    img_meta = img_test.ome_metadata
    pixel_size = metadata.utils.physical_pixel_sizes(img_meta)
    return pixel_size


def compute_zoom_coefs(img_path, zoom_dict, sampling):
    """
   Compute zoom coefficients based on pixel sizes extracted from an image and 
   store them in a dictionary.

   Parameters:
   - img_path (str): Path to the image file.
   - zoom_dict (dict): Dictionary to store zoom coefficients.

   Returns:
   - coefs (list): List containing zoom coefficients.
   - zoom_dict (dict): Updated dictionary containing zoom coefficients 
       corresponding to image paths.

   Notes:
   - This function calculates zoom coefficients based on the pixel sizes 
       extracted from the input image.
   - Zoom coefficients are computed as the ratio of the z-axis pixel size 
       to the y-axis pixel size, with x-axis assumed to have the same pixel 
       size as y-axis.
   - The computed coefficients are stored in the zoom_dict against the image 
       path.
   """
   
    pixel_size = get_pixel_size(img_path)
    z,y,x = pixel_size
    
    if (y == x):
        zoom = z/y
        coefs = np.array([zoom, 1, 1])
        zoom_dict[img_path] = [pixel_size, coefs, sampling, sampling * coefs]
        return sampling * coefs, zoom_dict
    
    else:
        print("!! different sizes in the x and y dimensions !!")
        return [1,1,1], zoom_dict


def read_mitochondria(path, zoom_dict, sampling):
    '''Read channel with mitochondria and zoom using given coefs'''
    czi = czifile.imread(path)
    czi_img = czi[0, 0, 0, 0, : , : , : , 0]
    zoom_coefs, zoom_d = compute_zoom_coefs(path, zoom_dict, sampling)
    img_zoom = ndim.zoom(czi_img, zoom_coefs)
    
    zoom_d[path].append(czi_img.shape)
    zoom_d[path].append(img_zoom.shape)
    
    return img_zoom, zoom_d


def read_cytoskelet(path, zoom_dict, sampling):
    '''Read channel with cytoskelet and zoom using given coefs'''
    czi = czifile.imread(path)
    czi_img = czi[0, 0, 1, 0, : , : , : , 0]
    zoom_coefs, zoom_d = compute_zoom_coefs(path, zoom_dict, sampling)
    img_zoom = ndim.zoom(czi_img, zoom_coefs)
    
    return img_zoom, zoom_d


def create_max_proj(image):
    
    proj = np.amax(image, axis=0)
    
    return proj


def diffuse_filter_3d(img_zoom, max_radius = 4, step = 1):
    
    '''
   Perform diffuse background removal using median filtering with varying radii.

   Parameters:
   - img_zoom (numpy.ndarray): 3D NumPy array representing the input image.
   - max_radius (int): Maximum radius of the footprint (ball) for median 
       filtering (default is 4).
   - step (int): Step size for varying the radius (default is 1).

   Returns:
   - min_med_img (numpy.ndarray): 3D NumPy array representing the image after 
       diffuse background removal.

   Notes:
   - This function performs diffuse background removal by applying median 
       filtering with varying radii.
   - The radius of the footprint (ball) for median filtering starts from 1 
       and increases up to (max_radius - 1) with the specified step size.
   - The original image is also included in the filtering process.
   - The minimum value across all filtered images is computed to obtain
       the final result.

    
    '''
    # Perform median filtering with varying radius of the footprint (ball)
    imgs_med =[]
    
    for radius in range(1, max_radius, step):
        diff_ball = morphology.ball(radius)
        imgs_med.append(ndim.median_filter(img_zoom, footprint=diff_ball))
        
    # Add original img to the stack
    imgs_med.append(img_zoom) 
    imgs_med_stack = np.stack((imgs_med))   
    
    # Get min through 3d imgs 
    min_med_img = np.min(imgs_med_stack, axis=0)
    
    return min_med_img


def img_preprocessing(source_folder, 
                      dest_folder, 
                      actions:dict,  
                      save_as = 'both',
                      img_type = 'czi', 
                      sampling = 1):
    '''Applies a set of image processing actions to all CZI files (default) in 
        the specified source folder, and saves the resulting images as NumPy 
        arrays or TIF files in the specified destination folder.
    
    Args:
        source_folder (str): The path to the folder containing the 
                                input CZI files (default).
        dest_folder (str): The path to the folder where the output images will 
                                be saved.
        actions (dict): A dictionary of image processing actions, where the 
                        keys are strings representing the name of each action, 
                        and the values are corresponding functions that accept
                        an image file path and a list of zoom coefficients as 
                        arguments and return a processed image.
        save_as (str): Specifies the format in which the output images will be
                        saved. Can be 'numpy', 'tif', or 'both' (default='both').
        sampling: This variable is used in function "compute_zoom_coefs" - 
                    calculated coefs are multiplied by this factor to produce 
                    smaller image, value should be from range (0, 1), 
                    default = 1.
    
    Returns:
        None
    '''
    zoom_dict = {}
    
    for abs_path in glob.glob( os.path.join( source_folder, f"**/*.{img_type}" ), recursive=True ):
        print(f"processing: {abs_path}")
        rel_path = os.path.relpath( abs_path, source_folder )
        dest_path = os.path.join( dest_folder, rel_path )
        os.makedirs( os.path.dirname( dest_path ), exist_ok=True ) 
    
        for name, action in actions.items():
            
            if name == 'mit' or name == 'cyt':
                img_res, zoom_dict = action(abs_path, zoom_dict, sampling) 
            else:
                img_res = action(abs_path) 
                
            new_np_name = dest_path.replace('SIM².czi', f'_{name}.npy')
            new_tif_name = dest_path.replace('SIM².czi', f'_{name}.tif')
            
            if save_as == 'numpy':
                np.save(new_np_name, img_res)
            elif save_as == 'tif':
                tifffile.imsave(new_tif_name, img_res)
            elif save_as == 'both':
                np.save(new_np_name, img_res)
                tifffile.imsave(new_tif_name, img_res)

    with open(os.path.join(dest_folder, "zoom_info.txt"), 'w', encoding="utf-8") as f:
        for name, vals in zoom_dict.items():
            print(f'{name}: {vals}\n')
            f.write(f'{name}: {vals}\n')
            
    
    
def np_arr_analysis(source_folder, 
                    dest_folder, 
                    actions:list, 
                    input_img_type = 'mit', 
                    save_as = 'numpy'):
    ''' 
    Perform analysis on NumPy arrays given in a dictionary "actions" and save 
    the results as new NumPy arrays or TIFF images.

    Parameters:
    - source_folder (str): Path to the folder containing the input NumPy arrays.
    - dest_folder (str): Path to the folder where the results will be saved.
    - actions (list): A list of tuples containing action names and corresponding 
        functions.
    - input_img_type (str): Type of input image files to consider (default is 'mit').
    - save_as (str): File format to save the processed images ('numpy' or 'tif').
        Default is 'numpy'.

    Returns:
    - None

    Notes:
    - The function iterates over all NumPy arrays in the source_folder whose 
        filenames contain input_img_type.
    - It applies each action specified in the actions dictionary to the loaded
        NumPy array in order in the dictionary.
    - The output of each action is set as the input for the following action in 
        the list.
    '''
    
    # Iterate over all NumPy array files in the source folder
    for abs_path in glob.glob( os.path.join( source_folder, f'*{input_img_type}.npy' ), recursive=True ):
        print(f"processing: {abs_path}")
        
        # Get the relative path of the file (image name)
        rel_path = os.path.relpath( abs_path, source_folder )
        # Create the destination path for saving processed arrays
        dest_path = os.path.join( dest_folder, rel_path )
        os.makedirs( os.path.dirname( dest_path ), exist_ok=True ) 
        
        # Load input image
        input_img = np.load(abs_path)
        
        # # JUST FOR TESTING, SMALLER SAMPLING !!!!!
        # print(input_img.shape)
        # input_img = input_img[::3, ::3, ::3]
        # print(input_img.shape)
        
        # Set input to the function (action)
        func_input = input_img
        
        # Iterate over each action specified in the actions list
        for action in actions:
            
            try:
                # Check if input for the current action exists
                func_input.shape
                
            except AttributeError:
                # If input does not exist, print an error message and break out 
                # of the loop through actions
                print('actions cannot continue, there is no input' )
                break
                    
            # Get action name and function from the current action tuple
            action_name, func = action
            print(f'Performing: {action_name}')
    
            if action_name == 'cut_roi_3d':
                save_as = None
                func(func_input, rel_path)
            else: 
                # Perform action
                img_res = func(func_input)
                
                
            # Save the processed image based on the specified save_as format
            if save_as == None:  
                pass
            
            elif save_as == 'numpy':   
                if type(img_res) == dict:
                    # If the result is a dictionary, save it as an .npz file
                    new_name = dest_path.replace('.npy', f'_{action_name}.npz')
                    np.savez(new_name, **img_res)
                    
                    print(f"Cannot continue analysis, last action ({action_name}) returned dictionary")
                    # Set input to None as it cannot be continued further
                    func_input = None
                
                else: 
                    # Save the result as a .npy file
                    new_name = dest_path.replace('.npy', f'_{action_name}.npy')
                    np.save(new_name, img_res)
                    
                    # Set output as a new input 
                    func_input = img_res
                    

            elif save_as == 'tif':
                new_name = dest_path.replace('.npy', f'_{action_name}.tif')
                tifffile.imsave(new_name, img_res)
                
                # Set output as a new input 
                func_input = img_res
                    
            
            

#OLD
def old_load_npy(folder_path, im_type = 'mit', return_type= 'list', name= None):
    '''
    
    Load npy files from a folder and return them in a list or dictionary.

    Parameters:
    - folder_path (str): The path to the folder containing the npy files.
    - im_type (str): The type of npy files to load (options: 'mit', 'cyt', 
        'thr', 'skelet', 'eq', 'diffused', 'dict_segm'). Default is 'mit'.
    - return_type (str): The desired return type ('list' or 'dictionary'). 
        Default is 'list'.
    - name (str, optional): In case of im_type == 'dict_segm', if specified, 
        load a specific named file from the dictionary (e.g., 'skelet'). 
        Default is None.

    Returns:
    - list or dictionary: A list or dictionary containing the loaded npy files.
    
    '''
    # Initialize list and dictionary to store loaded images
    imgs = []
    imgs_dict = {}
    
    # Load npy files based on specified im_type and return_type
    if name is None:
        # If name is not specified, load all npy files of the given im_type
        for file_path in glob.glob(os.path.join(folder_path, f'*{im_type}.npy')):
            print(file_path)
            img = np.load(file_path, allow_pickle=True)
            imgs.append(img)
            imgs_dict[file_path] = img
    else:
        # if the name of the file from the dictionary that was load is specified, e.g., name = 'skelet'
        print(f'{name}')
        for dict_name in glob.glob(os.path.join(folder_path, f'*{im_type}.npy')):
            print(dict_name)
            dict_ = np.load(dict_name, allow_pickle=True)
            dict_ = dict_.item()
            if return_type == 'list':
                imgs.append(dict_[name])
            elif return_type == 'dictionary':
                current_img_name = os.path.basename(dict_name)
                print(current_img_name)
                imgs_dict[current_img_name] = dict_[name]
    
    if return_type == 'list':
        return imgs
    elif return_type == 'dictionary':
        return imgs_dict


def load_npy(folder_path: str, 
             im_type: str = 'mit', 
             return_type: str = 'list'):
    '''
    Load npy files from a folder and return them in a list or dictionary.

    Parameters:
    - folder_path (str): The path to the folder containing the npy files.
    - im_type (str): The type of npy files to load (options: 'mit', 'cyt', 
                                                    'thr', 'skelet', 'eq', 
                                                    'diffuse'). 
                    Default is 'mit'.
    - return_type (str): The desired return type ('list' or 'dictionary'). 
        Default is 'list'.

    Returns:
    - list or dictionary: A list or dictionary containing the loaded npy files.
    '''

    # Initialize the variable to store loaded images
    loaded_images = {}

    # Load npy files based on specified im_type
    for file_path in glob.glob(os.path.join(folder_path, f'*{im_type}.npy')):
        print(file_path)
        img = np.load(file_path, allow_pickle=True)
        
        loaded_images[file_path] = img

    # Return loaded images based on return_type
    if return_type == 'list':
        return list(loaded_images.values())
    
    elif return_type == 'dictionary':
        return loaded_images


def load_npz(folder_path: str, 
             im_type: str = 'dict_segm', 
             return_type: str = 'list', 
             name: str = None):
    '''
    Load npz files from a folder and return them in a list or dictionary.

    Parameters:
    - folder_path (str): The path to the folder containing the npz files.
    - im_type (str): The type of npz files to load, default is 'dict_segm'.
    - return_type (str): The desired return type ('list' or 'dictionary'). 
        Default is 'list'.
    - name (str, optional): If specified, used as a key to load only specific 
        image from the dictionary save in npz (e.g., 'skelet'). If None, whole 
        dictionary is loaded.

    Returns:
    - list: List of images chosen form the npz based on the parameter name. If 
            name is None, returned list contains full dictionaries from the 
            specified folder.
    - dictionary: Dictionary containing file paths as a key and specified image
                    (based on the name) as a value. If name is None, values are
                    full dictionaries saved in the specified folder.
    '''

    # Initialize the variable to store loaded images
    loaded_images = {}

    # Load npy files based on specified im_type
    for file_path in glob.glob(os.path.join(folder_path, f'*{im_type}*.npz')):
        print(f'loading: {file_path}')
        
        with np.load(file_path, mmap_mode='r') as loaded:
                
            if name is None:
                # Load full dictionary 
                loaded_dict = {key: loaded[key] for key in loaded}
                loaded_images[file_path] = loaded_dict
                
            else:
                # If name is specified, load specified item
                print(f'{name}')
                image = loaded[name]
                loaded_images[file_path] = image

    # Return loaded images based on return_type
    if return_type == 'list':
        return list(loaded_images.values())
    
    elif return_type == 'dictionary':
        return loaded_images




def equalize(img, radius = 10):
    '''
    Equalize image using local histogram. Mask, where equalization is performed, 
    is given by Otsu threshold.
    '''
    # Generate a mask using Otsu's thresholding method (to exclude background)
    mask = img >= filters.threshold_otsu(img)
    
    # Create a ball-shaped footprint for local histogram equalization
    footprint = morphology.ball(radius)
    
    # Perform local histogram equalization on the image with the specified footprint and mask
    img_eq = filters.rank.equalize(img, footprint=footprint, mask = mask)
    
    return img_eq



def apply_ridge_filter_skeletonize(img_eq, 
                       filter_type = None, 
                       sigmas = range(1,3), 
                       threshold = filters.threshold_triangle, 
                       remove_s_objects = 100 ):
    
    '''Applies a ridge filter to an input image and returns a dictionary 
        containing the intermediate and final results.
    
    Args:
        img_eq (ndarray): A 3D grayscale input image.
        filter_type (function): The ridge filter function to use. 
                                Can be 'frangi', 'meijering', 'sato', 'hessian'.
                                If filter_type is None, ridge filtering is skipped.
        sigmas (range): Sigmas used as scales of filter, 
                        i.e., np.arange(scale_range[0], 
                                        scale_range[1], 
                                        scale_step (default=range(1,5)).
        threshold (function): The thresholding function to use 
                                (default=filters.threshold_triangle - suitable 
                                 for confocal). Also possible to use 
                                filters.threshold_otsu - more suitable for superres.
        remove_s_objects (int): The minimum size (in pixels) of small objects 
                                to be removed from the binary image (default=100).
    
    Returns:
        A dictionary containing the following items:
            - 'f_img': The filtered image obtained from applying the specified 
                        ridge filter function to the input image.
            - 'thr_f': The thresholded binary image obtained from applying the 
                        specified thresholding function to 'f_img'.
            - 'clear': The binary image obtained after removing small objects 
                        from 'thr_f'.
            - 'closed': The binary image obtained after performing a binary 
                        closing operation on 'clear'.
            - 'skelet': The skeletonized version of 'closed'.
    '''
    
    # Apply ridge filter
    if filter_type is not None:
        print('apply filter')
        f_img = filter_type(img_eq, sigmas= sigmas, black_ridges = False)
    else:
        print('no ridge filter')
        f_img = img_eq
        
    
    # Thresholding
    thr_f = f_img >= threshold(f_img)
    
    # Remove small objects
    print('remove small objects')
    clear = morphology.remove_small_objects(thr_f, remove_s_objects)
    
    
    # Binary closing operation (semgented mask)
    closed = ndim.binary_closing(clear, structure = morphology.ball(3)) 
    
    # Skeletonization
    print('perform skeletonization')
    skelet = skeletonize(closed)
    
    
    # Return dictionary containing intermediate and final results
    return {'f_img': f_img , 
            'thr_f': thr_f, 
            'clear': clear,
            'closed': closed,
            'skelet': skelet }


def tifs2npy(source_folder):
    '''
    Read tif files form source folder.
    '''
    img_dict = {}
    for abs_path in glob.glob( os.path.join( source_folder, "*.tif" ), recursive=True ):
        head, tail = os.path.split(abs_path)
        img = tifffile.imread(abs_path)
        img_dict[tail] = img
        
    return img_dict

            
            
def read_and_analyse_tifs(source_folder, dest_folder, actions:dict):
    '''Read tif files form source folder, apply actions form actions dictionary 
        and save results as npy into destination folder.
    '''
    
    for abs_path in glob.glob( os.path.join( source_folder, "**/*.tif" ), recursive=True ):
        rel_path = os.path.relpath( abs_path, source_folder )
        dest_path = os.path.join( dest_folder, rel_path )
        os.makedirs( os.path.dirname( dest_path ), exist_ok=True ) 
        
        for name, action in actions.items():
            img = tifffile.imread(abs_path)
            img_res = action(img) 
            new_np_name = dest_path.replace('.tif', f'_{name}.npy')
            np.save(new_np_name, img_res)











