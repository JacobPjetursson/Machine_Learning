#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:11:38 2017

@author: Jacob
"""
import numpy as np
from matplotlib.pyplot import bar, figure, subplot, plot, hist, title, show
from scipy.stats.kde import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import dataSetup
X = dataSetup.numbersData.values
X = stats.zscore(X)
"""
# OUTLIER DETECTION
# Compute kernel density estimate
kde = gaussian_kde(X.ravel(), 'silverman')

scoresKDE = kde.evaluate(X.ravel())
idxKDE = scoresKDE.argsort()
scoresKDE.sort()

print('The index of the lowest density object: {0}'.format(idxKDE[0]))

# Plot kernel density estimate
figure()
bar(range(20),scoresKDE[:20])
title('Outlier score KDE')
show()

"""
# Number of neighbors
K = 200

# x-values to evaluate the KNN
xe = np.linspace(-10, 10, 100)
x = np.linspace(-10, 10, 50)
N = 1000; M = 2
X = np.empty((N,M))
m = np.array([1, 3, 6]); s = np.array([1, .5, 2])
c_sizes = np.random.multinomial(N, [1./3, 1./3, 1./3])
for c_id, c_size in enumerate(c_sizes):
    X[c_sizes.cumsum()[c_id]-c_sizes[c_id]:c_sizes.cumsum()[c_id],:] = np.random.normal(m[c_id], np.sqrt(s[c_id]), (c_size,M))

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(np.matrix(xe).T)

# Compute the density
#D, i = knclassifier.kneighbors(np.matrix(xe).T)
knn_density = 1./(D.sum(axis=1)/K)

# Compute the average relative density
DX, iX = knn.kneighbors(X)
knn_densityX = 1./(DX[:,1:].sum(axis=1)/K)
knn_avg_rel_density = knn_density/(knn_densityX[i[:,1:]].sum(axis=1)/K)

# Plot KNN density
figure(figsize=(6,7))
subplot(2,1,1)
hist(X,x)
title('Data histogram')
subplot(2,1,2)
plot(xe, knn_density)
title('KNN density')
# Plot KNN average relative density
figure(figsize=(6,7))
subplot(2,1,1)
hist(X,x)
title('Data histogram')
subplot(2,1,2)
plot(xe, knn_avg_rel_density)
title('KNN average relative density')

show()
