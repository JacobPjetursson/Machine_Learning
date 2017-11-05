#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:20:42 2017

@author: Jacob
"""
import numpy as np
import dataSetup
from matplotlib.pyplot import boxplot, plot, hold, xticks, figure, subplot,title, hist, xlabel, ylim, show
import matplotlib.pyplot as plt # Python defacto plotting library
import seaborn as sns # More snazzy plotting library

numbersData = dataSetup.numbersData
X = dataSetup.X
M = dataSetup.M
attributeNames = dataSetup.attributeNames
# BOXPLOT               
figure(figsize=(12,7))
title('Movies: Boxplots')
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    boxplot(X[:,i], sym='k')
    xlabel(attributeNames[i])

#BOXPLOT BUDGET
figure(figsize=(6,4))
boxplot(X[:, 3], sym='k')
xlabel("Budget")

figure(figsize=(6,4))
boxplot(X[:, 0], sym='k')
xlabel("Duration")
#xticks(range(1,M+1), attributeNames, rotation=45)

# HISTOGRAMS OF ALL ATTRIBUTES
figure(figsize=(12,7))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    
# SCATTER PLOT MATRIX
# Set style of scatterplot
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
# Create scatterplot of dataframe
sns.set(style="ticks")
#pd.plotting.scatter_matrix(numbersData, alpha=0.2, figsize=(12,12), diagonal='kde')
sns.pairplot(numbersData, size=1.5)

# CORRELATION MATRIX:
f, ax = plt.subplots(figsize=(10, 8))
plt.title('Pearson Correlation of movie data')
sns.heatmap(numbersData.astype(float).corr(),linewidths=0.2,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)