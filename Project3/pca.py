# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:40:36 2017

@author: Meowasaurus
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:20:42 2017

@author: Jacob
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from dataSetup import *
X = stats.zscore(X)
pca = PCA(n_components=2)
x_2d = pca.fit_transform(X_std)

#print(x_2d.shape)
"""
plt.figure(figsize=(10,7))
plt.scatter(x_2d[:,0],x_2d[:,1], c='goldenrod', alpha=0.5)
plt.ylim(-5,7)
plt.show
"""
#print(X_std)

#numbersData.plot(y= 'imdb_score', x ='gross',kind='hexbin',gridsize=35, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Gross',figsize=(12,8))

