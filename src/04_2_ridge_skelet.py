from skimage.filters import meijering, sato, frangi, hessian, threshold_triangle
from skimage import morphology 
import scipy.ndimage as ndim
import numpy as np
import tifffile
import os
import glob
import utils.utils3d as ut3d

def update_filter(fn):
    def wrapper(*args, **kwargs):
        def specified_filter(img):
            return fn(img, *args, **kwargs)
        return specified_filter
    return wrapper

def no_filter():
    pass

@update_filter
def new_apply_ridge_filter_skeletonize(img_eq, 
                       filter_type = None, 
                       sigmas = range(1,3), 
                       threshold = threshold_triangle, 
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
        filter_type = no_filter
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
    skelet = morphology.skeletonize(closed)
    
    filter_name = filter_type.__name__
    
    # Return dictionary containing intermediate and final results
    return {f'f_img_{filter_name}': f_img , 
            'thr_f': thr_f, 
            'clear': clear,
            'closed': closed,
            'skelet': skelet }



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
                    
        

with open('04_2_paths.txt', 'r') as f:
    paths = f.readlines()
    source_folder = paths[0].strip()
    dest_folder = paths[1].strip()



actions = [
            ('dict_segm', new_apply_ridge_filter_skeletonize(filter_type = frangi)) # do not forget to set correct ridge filter ('frangi', 'meijering', 'sato', 'hessian' or None)
            ]


np_arr_analysis(source_folder, dest_folder, actions, input_img_type = 'eq')




# filters = [frangi, meijering, sato, hessian, None]
# for filter in filters:
#     if filter == None:
#         actions = [ 
#                 (f'dict_segm_', new_apply_ridge_filter_skeletonize(filter_type = filter)), # do not forget to set correct ridge filter ('frangi', 'meijering', 'sato', 'hessian' or None)
#             ]
#     else:
#         actions = [ 
#                     (f'dict_segm_{filter.__name__}', new_apply_ridge_filter_skeletonize(filter_type = filter)), # do not forget to set correct ridge filter ('frangi', 'meijering', 'sato', 'hessian' or None)
#                 ]

#     np_arr_analysis(source_folder, dest_folder, actions, input_img_type = 'eq')



# # see results
# import numpy as np
# import napari
# import glob
# import os.path

# file_path = dest_folder + "\show_imgs.txt"
# viewer = napari.Viewer(ndisplay=3)
# with open(file_path, 'r', encoding="utf-8") as file:
#       # Iterate over each line in the file
#       for line in file:

#           for im_path in glob.glob( os.path.join( dest_folder, line ).replace('\n', '') + '*'):
#               print(im_path)
              
#               if 'segm' in im_path:
#                   im_dict = np.load(im_path)
#                   for name, im in im_dict.items():
#                             viewer.add_image(im, name=f'{name}')
#               else:
#                   im = np.load(im_path)
#                   viewer.add_image(im, name = im_path)
  
              

# napari.run()  # use in VS code                    