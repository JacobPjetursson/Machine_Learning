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
from dataSetup import *
from scipy import stats
from sklearn.decomposition import PCA


pca = PCA(n_components=1)
x = pca.fit_transform(X_std)
"""
plt.figure(figsize=(10,7))
plt.scatter(x[:,0],x[:,1], c='goldenrod', alpha=0.5)
plt.ylim(-5,7)
plt.show
"""
#print(X_std)

#numbersData.plot(y= 'imdb_score', x ='gross',kind='hexbin',gridsize=35, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Gross',figsize=(12,8))

