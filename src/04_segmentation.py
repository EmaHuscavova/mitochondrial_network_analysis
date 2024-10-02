# # -*- coding: utf-8 -*-
# """
# Created on Tue Nov  7 11:35:34 2023

# @author: emahu
# """


import utils.utils3d as ut3d

# folder with 3d single cells
with open('04_paths.txt', 'r') as f:
    paths = f.readlines()
    source_folder = paths[0].strip()
    dest_folder = paths[1].strip()

# diffuse filter and equalize local histogram
actions = [
           ('diffuse', ut3d.diffuse_filter_3d), 
           ('eq', ut3d.equalize)  
           ]


ut3d.np_arr_analysis(source_folder, dest_folder, actions)



# # see results
# import numpy as np
# import napari
# import glob
# import os.path

# file_path = r"..\see_results.txt"
# viewer = napari.Viewer(ndisplay=3)
# with open(file_path, 'r', encoding="utf-8") as file:
#       # Iterate over each line in the file
#       for line in file:

#           for im_path in glob.glob( os.path.join( dest_folder, line ).replace('\n', '') + '*'):
#               # print(im_path)
              
#               if 'segm' in im_path:
#                   im_dict = np.load(im_path)
#                   for name, im in im_dict.items():
#                             viewer.add_image(im, name=f'{name}')
#               else:
#                   im = np.load(im_path)
#                   viewer.add_image(im, name = im_path)
  
              

# napari.run()  # use in VS code                    