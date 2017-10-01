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

fields= ['budget','gross','genres', 'imdb_score', 'num_voted_users']
numberFields = ['budget','gross', 'imdb_score', 'num_voted_users']
movie = pd.read_csv('movie.csv', encoding="latin-1",usecols=fields)
numbersData = pd.read_csv('movie.csv', encoding="latin-1",usecols=numberFields)
numbersData = numbersData.fillna(value=0, axis=1) # TODO - Wrong to swap with 0


X = numbersData.values
# Data Normalization
X_std = StandardScaler().fit_transform(X)

pca = PCA(n_components=4)
x_4d = pca.fit_transform(X_std)

plt.figure(figsize=(10,7))
plt.scatter(x_4d[:,0],x_4d[:,1], c='goldenrod', alpha=0.5)
plt.ylim(-10,30)
plt.show

#print(X_std)

#numbersData.plot(y= 'imdb_score', x ='gross',kind='hexbin',gridsize=35, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Gross',figsize=(12,8))

