# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:43:41 2024

@author: emahu
"""

# mitochondrial morphology

import numpy as np
import napari
import utils.utils3d as ut3d
import os.path
import matplotlib.pyplot as plt



def calc_volume(img, px_size):
    '''
    Calculate total volume of the mitochondrial network in the physical units 
    same as the input.

    '''
    # [um]
    z, y, x = px_size
    voxel = z * y * x
    # print(np.sum(img), voxel, px_size)
    volume = np.sum(img) * voxel

    return float(volume)


def find_image_px_size(file_path: str, search_image_name: str):
    '''
    Open a text file, iterate over the lines, find line that contains 
    search_image_name, and print the string that starts with 
    "(PhysicalPixelSizes" and ends with "),".

    Parameters:
    - file_path (str): The path to the text file.
    - search_string (str): The string to search for in the text file.

    Returns:
    - Returns pixel sizes from the .txt file for specified image.
    '''
    try:
        # Open the text file in read mode
        with open(file_path, 'r', encoding="utf-8") as file:
            # Iterate over each line in the file
            for line in file:
                
                if search_image_name in line:
                    # Find the substring that starts with "(PhysicalPixelSizes"
                    # and ends with "),"
                    start_index = line.find('(PhysicalPixelSizes')
                    end_index = line.find('),')
                    
                    if start_index != -1 and end_index != -1:
                        sizes = line[start_index : end_index + 2]
                        
                        s_index = sizes.find('Z')
                        e_index = sizes.find(',')
                        z = float(sizes[s_index + 2 : e_index - 1])
                        sizes = sizes[e_index + 1 : ]
                        
                        s_index = sizes.find('Y')
                        e_index = sizes.find(',')
                        y = float(sizes[s_index + 2 : e_index - 1])
                        sizes = sizes[e_index + 1 : ]
                        
                        s_index = sizes.find('X')
                        e_index = sizes.find(',')
                        x = float(sizes[s_index + 2 : e_index - 1])
                        
                        return z, y, x
                        
        
                        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")


###############################################################################

# folder with filtered segmented cells
with open('05_2_paths.txt', 'r') as f:
    paths = f.readlines()
    source_folder = paths[0].strip()
    dest_folder = paths[1].strip()
    # define path to the zoom_info.txt file (produced by 01_preprocessing.py)
    info_path = paths[2].strip()

# load semgented images
segmented = ut3d.load_npz(source_folder, 
                          im_type = 'dict_segm', 
                          name= 'clear', 
                          return_type= 'dictionary')

# # see images
# viewer = napari.Viewer(ndisplay=3)
# for name, im in segmented.items():
#     print(np.max(im))
#     viewer.add_image(im, name=f'{name}')




total_volume = dict()

for img_path, img in segmented.items():
    
    # find only base name of the image
    img_name_full = os.path.basename(img_path)
    pos = img_name_full.find('_mit')
    img_name = img_name_full[:pos]
    print(img_name)
    
    # find out pixel size for the specific image 
    px_size = find_image_px_size(info_path, img_name)
    
    # calculate volume of the network in um3
    volume = calc_volume(img, px_size)
    print(volume)
    print(type(volume))
    total_volume[img_name_full] = volume
    

# contains TOTAL VOLUME PER IMAGE
np.savez(os.path.join(dest_folder, 'total_volume_per_cell_full.npz'), **total_volume)


###############################################################################
    
# VOLUME PER CONNECTED COMPONENT
import utils.utils3d as ut3d
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import napari
import numpy as np
import matplotlib.pyplot as plt


def vol_components(cropped_name, image, info_path):
    '''
    Calculate volume per connected component.
    Return dict: keys - label of component, values - volume of the component [um3].
    '''
    vols = {}
    
    x, y, z = find_image_px_size(info_path, cropped_name)
    voxel = z * y * x
    
    labels = np.unique(image)
    
    for lab in labels:
        # background is -1
        if lab != -1:
            temp_im = image == lab
            vols[lab] = np.sum(temp_im) * voxel
            print(lab)
    
    return vols



# run this part separately from the overall volume computation
# folder with filtered segmented cells
dest_folder = r"..\filtered\full"

# load semgented images
segmented = ut3d.load_npz(dest_folder, 
                          im_type = 'dict_segm', 
                          name= 'clear', 
                          return_type= 'dictionary')

# define path to the zoom_info.txt file (produced by 01_preprocessing.py)
info_path = r"..l\zoom_info.txt"


# read dictionary with labeled skeletons from 05_skelet_analysis
folder_path = r"..\skelet_analysis"


#['MCF', 'MDA','4T'] 

tr = 'MCF'
labeled = ut3d.load_npz(folder_path, im_type= tr)
labeled1 = labeled[0]

# # see images
# viewer = napari.Viewer(ndisplay=3)
# for name, im in labeled.items():
#     viewer.add_image(im, name=f'{name}')
#     viewer.add_image(segmented[name.replace('/', '\\')], name=f'{name}') # POZOR LOMITKA ?!?
    
    
# viewer = napari.Viewer(ndisplay=3)

labeled_components = {}
vol_per_comp = {}

# label each component in the volume image with the label of the corresponding skeleton
# watershed with labeled components
for name, labeled_img in labeled1.items():
    volume_img = segmented[name.replace('/', '\\')]
    # viewer.add_image(volume_img, name='vol')
    
    # labeling of the components with watershed and labels from skeletons
    print('calculating distance')
    distance = ndi.distance_transform_edt(volume_img)
    # viewer.add_image(distance, name='dist')
    # viewer.add_image(-distance, name='-dist')
    
    # viewer.add_image(labeled_img, name='labeled skelet')
    
    # set background to 0, important for watershed algorithm
    labeled_img = labeled_img + 1
    print(f'watershed: {name}')
    res_img = watershed(-distance, labeled_img, mask= volume_img)
    
    # return to the consistent labeling, background is -1, labeling from 0
    res_img = res_img - 1
    # viewer.add_image(res_img, name=f'{name}')
    
    # add labeled components (volume) to the dictionary
    labeled_components[name] = res_img
    
    # get basename of the file
    pos = name.find('63x')
    only_img_name = name[pos:]
    print(only_img_name)
    
    # adjusto to use function find_image_px_size
    pos_end = name.find('_mit')
    cropped_name = only_img_name[pos:pos_end]
    print(cropped_name)
    
    # create dict that contains volume per connected component
    vols = vol_components(cropped_name, res_img, info_path)

    # volumes of components per image
    vol_per_comp[only_img_name] = list(vols.values())

    
    
# # save labeled images
# np.savez(rf'D:\mitochondria_results\Jirka\superres_BCV-141\labeled_components_imgs_{tr}_full.npz', **labeled_components)

# np.savez(rf'D:\mitochondria_results\Jirka\superres_BCV-141\vol_per_comp_{tr}_full.npz', **vol_per_comp)




# # see results
# folder_path = r'D:\mitochondria_results\Jirka\superres_BCV-141'

# labeled_components = ut3d.load_npz(folder_path, im_type = 'labeled_components_sample_small')                      
# res = labeled_components[0]

# labeled_skelet = ut3d.load_npz(folder_path, im_type= 'labeled_components_sample_small')
# skelet = labeled_skelet[0]

# viewer = napari.Viewer(ndisplay=3)
# for name, im in res.items():
#     viewer.add_image(im, name=f'{name}')