# Mitochondria analysis


This analysis is suitable for evaluating mitochondrial networks and assessing multiple parameters derived from the skeleton and volume of the network. This analysis is suitable for three-dimensional images from confocal of super-resolution microscopy.

For analysis of 3D images of the mitochondrial network follow these steps:
- run **01_preprocessing.py**, which will determine the voxel's physical size in the image. It zooms (ndim.zoom()) the image to have uniform sampling on all axes, and returns preprocessed images and .txt file with zoom info.

Note: If there are multiple cells in the image, you can manually outline each one and create a separate mask for each cell.

Selecting cells manually:
- run **02_max_projection.py** with preprocessed images to create max projection tifs.
- using the drawing app (a custom PyQt interface is being developed), outline each cell in a separate mask, pay attention to the folder paths and types of images (tif, png, ...); note: pixels under the red marking will be included in the resulting image.
- run **03_single_cell_3d.py** which cuts out a single cell based on the drawn mask, returns single cells and and fills the surrounding area with black pixels to maintain the original shape. 

If you skip manual cell selection, analysis is performed on the whole image, and cell count will be considered in the later part of the analysis (cell_count.py).

- run **04_segmentation.py** (more time-consuming part of the analysis), 3D images of single cells are filtered using diffuse background removal, then local histogram equalization is performed and ridge filter may be applied (note:frangi filter for confocal?, for superres None), results are saved as a dictionary that contains following items:
    - 'f_img': The filtered image obtained from applying the specified ridge filter function to the input image.
    - 'thr_f': The thresholded binary image obtained from applying the specified thresholding function to 'f_img'.
    - 'clear': The binary image obtained after removing small objects from 'thr_f'.
    - 'closed': The binary image obtained after performing a binary closing operation on 'clear'.
    - 'skelet': The skeletonized version of 'closed'.

- run **cell_count.py**, load "clear" images from the previous script, each  image will be visualized in the console, and user input will be required to manually count the cells.


<img src="https://github.com/EmaHuscavova/mitochondria/assets/125351151/de6df370-23d1-489e-80ab-6c59dfa814a0" width=50% height=50%>

- run **new_05_skelet_analysis.py** - performs analysis of skeleton using library [skan](https://skeleton-analysis.org/stable/), returns statistical overview of the skeleton, calculates the overall volume of the network [&micro;m&sup3], and volumes of the individual components [&micro;m&sup3].

![cell_mito_skelet+skelet](https://github.com/EmaHuscavova/mitochondria/assets/125351151/e05fc585-e529-4b41-bc28-069bcafc3082)



