# -*- coding: utf-8 -*-
"""
@author: emahu
"""

import utils.utils3d as ut3d

with open('02_paths.txt', 'r') as f:
    paths = f.readlines()
    source_folder = paths[0].strip()
    dest_folder = paths[1].strip()


actions = [ ('max_proj', ut3d.create_max_proj) ]

ut3d.np_arr_analysis(source_folder, 
                     dest_folder, 
                     actions, 
                     input_img_type = 'mit', # 'mit' 'cyt'
                     save_as = 'tif')



