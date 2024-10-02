# -*- coding: utf-8 -*-
"""

@author: emahu
"""

import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items

tab_5uM = pd.read_csv(r"..\segm\cell_stat_5uM.csv")
tab_20uM = pd.read_csv(r"..\segm\cell_stat_20uM.csv")

dest_folder = r"..\res"

import seaborn as sns
import matplotlib.pyplot as plt
import os

tab_5uM = tab_5uM.drop(columns=['Unnamed: 0'])
tab_20uM = tab_20uM.drop(columns=['Unnamed: 0'])

def connect_params(params, tab1, tab2):
    new = pd.DataFrame()
    if len(params) == 1:
        new['5uM'] = tab1[f'{params[0]}']
        new['20uM'] = tab2[f'{params[0]}']
    else:
        for param in params:
            new[f'{param} - 5uM'] = tab1[f'{param}']
            new[f'{param} - 20uM'] = tab2[f'{param}']
            
    return new


params = ['mean lenght of continuous component [um]'] 
mean_len = connect_params(params, tab_5uM, tab_20uM)

plt.figure(figsize=(8,8))
sns.boxplot(data=mean_len)
sns.swarmplot(data=mean_len, color='black')
plt.ylabel('[um]')
plt.xticks(rotation=45)
plt.title('Mean lenght of continuous component per cell')
# plt.savefig( os.path.join( dest_folder, "con_comp.png" ), bbox_inches = 'tight')



params = ['total length of mitochondria [um]']
total_len = connect_params(params, tab_5uM, tab_20uM)

plt.figure(figsize=(8,8))
sns.boxplot(data=total_len)
sns.swarmplot(data=total_len, color='black')
plt.ylabel('[um]')
plt.xticks(rotation=45)
plt.title('Total length of mitochondrial skeleton')
# plt.savefig( os.path.join( dest_folder, "total_len.png" ), bbox_inches = 'tight')



forked = connect_params(['occur of forked branches'], tab_5uM, tab_20uM)
forked = forked.mean()
forked5 = forked['5uM']*100
forked20 = forked['20uM']*100

params = ['occur of isolated cycles', 
          'occur of isolated branches']
occur = connect_params(params, tab_5uM, tab_20uM)
new = occur*100

plt.figure(figsize=(12,8))
sns.boxplot(data=new)
sns.swarmplot(data=new, color='black')
plt.ylabel('%')
plt.xticks(rotation=45)
plt.title('Occurece of the specific type of branch')
plt.text(-0.4,1.6,f'Occurence of the forked branches:\n 5uM ~ {forked5:.2f}% \n 20uM ~ {forked20:.2f}%' )
# plt.savefig( os.path.join( dest_folder, "occur.png" ), bbox_inches = 'tight')






