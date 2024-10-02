# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:05:10 2023

@author: emahu
"""

import utils.utils_skelet_analysis as utskel
import utils.utils3d as ut3d
import numpy as np
from skan import Skeleton, summarize
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
import napari
import matplotlib.pyplot as plt
import os

# filtering - dict_segm files

with open('05_paths.txt', 'r') as f:
    paths = f.readlines()
    source_folder = paths[0].strip()
    dest_folder = paths[1].strip()


# read skeleton dict for analysis
segmented_skeletons = ut3d.load_npz(source_folder, name= 'skelet', return_type= 'dictionary') 
# read dict of the same segmented cells for visual counting
segmented_closed = ut3d.load_npz(source_folder, name= 'closed', return_type= 'dictionary') 



res = pd.DataFrame()

# set parameter num_cells=1 if the there is only single cell per image
num_cells = None

for img_path, cells in segmented_skeletons.items():
    
    # if the number of cells is not know
    if  num_cells == None:
        # show max projection of the 3D semgented cells 
        proj = np.amax(segmented_closed[img_path], axis=0)
        plt.figure(figsize=(10,10))
        plt.imshow(proj)
        plt.show()
        
        # MANUALLY COUNT NUMBER OF CELLS
        while True:
                
            try:
                # count by eye number of cell
                cell_count = input('Count number of cells:')
                int(cell_count)
            except ValueError:
                print('Input is not valid, needs to be number (int).')
                continue
            
            else:
                break
                    
    # if there is one cell per image    
    else:
        cell_count = num_cells
    
    # analyse skeletons in the image
    df_skel_summ = summarize(Skeleton(cells, spacing=0.049))
    df_cell = utskel.connected_components_stat(df_skel_summ)
    
    
    # pos = img_path.find('63x')
    # img_name = img_path[pos:]
    img_name = os.path.basename(img_path)
    print(img_name)
    
    treatments = ['CTRL', 'IC10', 'IC25', 'IC50']
    group = [tr for tr in treatments if tr in img_name]
    
    df_cell['cell count'] = cell_count
    df_cell['img_path'] = img_path
    df_cell['img name'] = img_name
    df_cell['group'] = group

    res = pd.concat([res, df_cell], ignore_index=True)

res.to_csv(os.path.join(dest_folder, 'cell_stat_full.csv'))



cell_types = ['MCF', 'MDA', '4T'] 

for c_type in cell_types:
    print(c_type)
    
    labeled_paths = {}

    for img_path, cells in segmented_skeletons.items():

        if c_type in img_path:
     
            df_skel_summ = summarize(Skeleton(cells, spacing=0.049))
            # use for visualization of labeled connected components
            print('consistent paths')
            consist_paths = utskel.connected_component_label_3D(cells, df_skel_summ) 
        
            # consist_paths is image with labeled components accordind to the summarize Skeleton
            # background of the image is -1, components are labeled from 0
            labeled_paths[img_path] = consist_paths
 
   
    np.savez_compressed(os.path.join(dest_folder, rf'labeled_imgs_full_{c_type}_compressed.npz'), **labeled_paths)






