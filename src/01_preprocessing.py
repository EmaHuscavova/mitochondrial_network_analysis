# -*- coding: utf-8 -*-
"""
Read .czi files in source folder - read channel with mitochondria 
and save preprocessed 3D images as numpy.
 
@author: emahu
"""

import utils.utils3d as ut3d

with open('01_paths.txt', 'r') as f:
    paths = f.readlines()
    source_folder = paths[0].strip()
    dest_folder = paths[1].strip()

# Each action in the dictionary is performed on the inpunt image
actions = {'mit': ut3d.read_mitochondria} # 'cyt': ut3d.read_cytoskelet or 'mit': ut3d.read_mitochondria

ut3d.img_preprocessing(source_folder, dest_folder, actions, save_as = 'numpy', sampling= 1)


