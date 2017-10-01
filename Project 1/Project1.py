#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:20:42 2017

@author: Jacob
"""
import numpy as np
import pandas as pd
from matplotlib.pyplot import boxplot, xticks, figure, subplot,title, hist, xlabel, ylim, show
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # Principal Component Analysis module
from sklearn.cluster import KMeans # KMeans clustering 
import matplotlib.pyplot as plt # Python defacto plotting library
import seaborn as sns # More snazzy plotting library
from scipy.stats import zscore

fields= ['budget','gross','genres', 'imdb_score', 'num_voted_users', 'movie_title']
numberFields = ['budget','gross', 'imdb_score', 'num_voted_users']
movie = pd.read_csv('movie.csv', encoding="latin-1",usecols=fields)
numbersData = pd.read_csv('movie.csv', encoding="latin-1",usecols=numberFields)
numbersData = numbersData.fillna(value=0, axis=1) # TODO - Wrong to swap with 0
X = numbersData.values
# Data Normalization
X_std = StandardScaler().fit_transform(X)
attributeNames = numbersData.columns.values
M = len(attributeNames)

# CORRELATION:
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 8))
plt.title('Correlation of movie data')
sns.heatmap(numbersData.astype(float).corr(),linewidths=0.2,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)


# OUTLIERS
# TITLES OF OUTLIERS
for i in movie:
    for k in movie.T:
        if(i == 'budget'):
            if(movie[i][k] > 500000000):
                print(movie['movie_title'][k])
                
figure(figsize=(12,6))
title('Movies: Boxplots')
boxplot(X, sym='k')
xticks(range(1,M+1), attributeNames, rotation=45)

# OUTLIERS REMOVED
outlier_mask = (X[:,2]>500000000)
valid_mask = np.logical_not(outlier_mask)
X = X[valid_mask,:]
figure(figsize=(12,6))
title('Movies: Boxplots')
boxplot(X, sym='k')
xticks(range(1,M+1), attributeNames, rotation=45)
