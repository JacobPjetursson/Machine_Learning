#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:21:18 2017

@author: Jacob
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

fields= ['budget','gross','genres', 'imdb_score', 'num_voted_users', 'movie_title', 'duration', 'title_year']
numberFields = ['budget','gross', 'imdb_score', 'num_voted_users', 'duration', 'title_year']
movie = pd.read_csv('movie.csv', encoding="latin-1",usecols=fields)
numbersData = pd.read_csv('movie.csv', encoding="latin-1",usecols=numberFields)

# OUTLIERS
outliers = []
for i in movie:
    for k in movie.T:
        if(i == 'budget'):
            if(movie[i][k] > 300000000 or movie[i][k] < 0) :
                outliers.append(k)
        if(i == 'duration'):
            if(movie[i][k] > 360 or movie[i][k] < 0):
                outliers.append(k)
                
movie.drop(movie.index[outliers], inplace=True)
numbersData.drop(numbersData.index[outliers], inplace = True)

grossMean = numbersData['gross'].mean()
imdbMean = numbersData['imdb_score'].mean()
budgetMean = numbersData['budget'].mean()
votedUsersMean = numbersData['num_voted_users'].mean()
durationMean = numbersData['duration'].mean()
titleyearMean = numbersData['title_year'].mean()

# NaN is changed with the mean of gross
numbersData['gross'] = numbersData['gross'].fillna(value=grossMean)
numbersData['imdb_score'] = numbersData['imdb_score'].fillna(value=imdbMean)
numbersData['budget'] = numbersData['budget'].fillna(value=budgetMean)
numbersData['num_voted_users'] = numbersData['num_voted_users'].fillna(value=votedUsersMean)
numbersData['duration'] = numbersData['duration'].fillna(value=durationMean)
numbersData['title_year'] = numbersData['title_year'].fillna(value=titleyearMean)

X = numbersData.values

# DATA NORMALIZATION
X_std = StandardScaler().fit_transform(X)
attributeNames = numbersData.columns.values
M = len(attributeNames)