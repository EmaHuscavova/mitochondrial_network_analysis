# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:12:18 2023

@author: emahu
"""

from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
from scipy.ndimage import convolve
import numpy as np
from skimage.measure import label
from skimage.morphology import dilation

from skan import Skeleton, summarize
import skan.csr
import pandas as pd


def connected_component_label(skeleton_img, df_skeleton):
    """
    Return image with components labeled consistently with
    'skeleton-id' from skan DataFrame, background is set to -1.
    """
    paths = label(skeleton_img)
    paths = paths - 1 # set background to -1 and label 1 to 0
    
    skel_count = np.unique(df_skeleton['skeleton-id'])
    
    for i in skel_count:
        df = df_skeleton.loc[df_skeleton['skeleton-id'] == i]
        df = df.sample()
        y = df['image-coord-src-0']
        x = df['image-coord-src-1']
        l = paths[y, x]
        paths[paths == l] = i
        
    return paths


def connected_component_label_3D(skeleton_img, df_skeleton):
    """
    Return image with components labeled consistently with
    'skeleton-id' from skan DataFrame, background is set to -1.
    """
    paths = label(skeleton_img)
    paths = paths - 1 # set background to -1 and label from 0
    
    # solve inconsistencies in skimage.measure.label vs. Skeleton
    skel_ids = np.unique(df_skeleton['skeleton-id']) # skan
    labels = np.unique(paths) # skimage
    difference = [i for i in labels if i not in skel_ids ]
    print(len(skel_ids))
    
    # remove labels from image that are not in Skeleton 
    # algorithm in skimage.measure.label may be different from the skan lib ??
    for i in difference:
        paths[paths == i] = -1 
    
    for i in skel_ids:
        df = df_skeleton.loc[df_skeleton['skeleton-id'] == i]
        df = df.sample()
        z = df['image-coord-src-0']
        y = df['image-coord-src-1']
        x = df['image-coord-src-2']
        # print(z,y,x)
        l = paths[z,y,x]
        paths[paths == l] = i
        print(i)
        

    return paths


def show_cycles(df_skeleton, components):
    """
    Return image with components only if containing cycle, according to skan.
    """
    df_cycles = df_skeleton[df_skeleton['branch-type'] == 3]
    cycle_ids = df_cycles['skeleton-id'].unique()
    cycles_im = np.zeros(components.shape) - 1
    # print(df_cycles['skeleton-id'])
    
    for i in cycle_ids:
        cycles_im[components == i] = i

    return cycles_im



def conn_comp_len(comp_id, df_skeleton): # not working correctly??
    '''
    compute leght of one connected component
    '''
    temp = df_skeleton[ df_skeleton['skeleton-id'] == comp_id ]
    sum_tab = temp.sum()
    dst = sum_tab['branch-distance'] 
    
    return dst


def connected_components_stat(df_skeleton):
    """
    Return DataFrame - stat for one cell. 
    Return occurence DF of differen types of connected components.
    """
    df_cell_stats = pd.DataFrame() 
    
    n_branches = df_skeleton.shape[0]
    
    components_count = np.max(df_skeleton['skeleton-id']) + 1
    df_cell_stats['number of continuous components'] = [components_count]
    
    # len of continuous component
    comp_len = []
    
    ids = df_skeleton['skeleton-id'].values
    for i in np.unique(ids):
        comp_len.append(conn_comp_len(i, df_skeleton))
     
    # added later
    df_cell_stats['len per component [um]'] = [comp_len]
    
    df_cell_stats['mean lenght of continuous component [um]'] = np.mean(comp_len)
    
    df_cell_stats['total length of mitochondria [um]'] = df_skeleton['branch-distance'].sum(0)
    
    # find out number of isolated branches or isolated cycles in the cell
    df_cell_stats['number of isolated cycles'] = len(df_skeleton[ df_skeleton['branch-type'] == 3  ])
    df_cell_stats['number of isolated branches'] = len(df_skeleton[ df_skeleton['branch-type'] == 0  ])
    df_cell_stats['number of forked branches'] =  len(df_skeleton[ (df_skeleton['branch-type'] == 1) | (df_skeleton['branch-type'] == 2) ])
    
    # count occurence of specific type of branch
    df_cell_stats['occur of isolated cycles'] = len(df_skeleton[ df_skeleton['branch-type'] == 3  ])/n_branches
    df_cell_stats['occur of isolated branches'] = len(df_skeleton[ df_skeleton['branch-type'] == 0  ])/n_branches
    df_cell_stats['occur of forked branches'] =  len(df_skeleton[ (df_skeleton['branch-type'] == 1) | (df_skeleton['branch-type'] == 2) ])/n_branches
    
    return df_cell_stats



def components_len_df(df_skeleton, name): # not working correctly??
    """
    Return list of conponents' lenght
    """
    components = []
    print(np.max(df_skeleton['skeleton-id']))
    
    for i in range(np.max(df_skeleton['skeleton-id'])):
        
        com_len_i = conn_comp_len(i, df_skeleton)
        # len of connected component is 0 if the index does not exist in the dataframe (cycle counter is not matching with id-s in the df)
        if com_len_i != 0:
            components.append(com_len_i)
        else: 
            print(name)
    
    return components


