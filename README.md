# Mitochondria analysis


This analysis is suitable for evaluating mitochondrial networks and assessing multiple parameters derived from the skeleton and volume of the network. The images analyzed using this method were obtained from confocal and super-resolution microscopy. This analysis is suitable for three-dimensional images.

For analysis of 3D images of the mitochondrial network follow these steps:
- run **01_preprocessing.py**, which will determine the voxel's physical size in the image. It zooms (ndim.zoom()) the image to have uniform sampling on all axes, and returns preprocessed images and .txt file with zoom info.

In case there are multiple cells in the image, it is possible to manually outline them and for each cell create a separate mask. 

Selecting cells manually:
- run **02_max_projection.py** with preprocessed images to create max projection tifs
- using the drawing app (custom PyQt interface) outline each cell in a separate mask, pay attention to the folder paths and types of images (tif, png, ...); note: pixels under the red marking will be included in the resulting image
- run **03_single_cell_3d.py** cuts out a single cell based on the drawn mask, returns single cells and fills with black pixels to obtain the original shape 

If you skip manual cell selection, analysis is performed on the whole image, and cell count will be considered in the later part of the analysis (cell_count.py).

- run **04_segmentation.py** (more time-consuming part of the analysis), 3D images of single cells are filtered using diffuse background removal, then local histogram equalization is performed and ridge filter may be applied (note:frangi filter for confocal?, for superres None), results are saved as a dictionary that contains following items:
    - 'f_img': The filtered image obtained from applying the specified ridge filter function to the input image.
    - 'thr_f': The thresholded binary image obtained from applying the specified thresholding function to 'f_img'.
    - 'clear': The binary image obtained after removing small objects from 'thr_f'.
    - 'closed': The binary image obtained after performing a binary closing operation on 'clear'.
    - 'skelet': The skeletonized version of 'closed'.

- run **cell_count.py**, load "clear" images from the previous script, each image is visualized in the console, and input from the user is needed - manually counted cells


<img src="https://github.com/EmaHuscavova/mitochondria/assets/125351151/de6df370-23d1-489e-80ab-6c59dfa814a0" width=50% height=50%>

- run **new_05_skelet_analysis.py** - performs analysis of skeleton using library [skan](https://skeleton-analysis.org/stable/), returns statistical overview of the skeleton, calculates the overall volume of the network [um^3], and volumes of the individual components [um^3]

![cell_mito_skelet+skelet](https://github.com/EmaHuscavova/mitochondria/assets/125351151/e05fc585-e529-4b41-bc28-069bcafc3082)



