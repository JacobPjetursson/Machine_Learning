#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:11:38 2017

@author: Jacob
"""
import numpy as np
from matplotlib.pyplot import (figure, imshow, bar, title, xticks, yticks, cm,
                               subplot, show)
from scipy.stats.kde import gaussian_kde
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import dataSetup

X = dataSetup.numbersData.values
X = stats.zscore(X)
N,M = X.shape

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

### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
   print('Fold {:2d}, w={:f}'.format(i,w))
   density, log_density = gausKernelDensity(X,w)
   logP[i] = log_density.sum()
   
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(X,width)

# Sort the densities
i_gaus = (density.argsort(axis=0)).ravel()
density = density[i_gaus].reshape(-1,)

# Plot density estimate of outlier score
figure(1)
bar(range(20),density[:20])
title('Density estimate')

print("\n")
print("Printing 20 lowest Gausian density points")
for j in range(0, 20):
    print("Index: " + str(i_gaus[j]) + "   -   Density: " + str(density[j]))

print("\n")
### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

density = 1./(D.sum(axis=1)/K)

# Sort the scores
i_KNN = density.argsort()
density = density[i_KNN]

# Plot k-neighbor estimate of outlier score (distances)
figure(3)
bar(range(20),density[:20])
title('KNN density: Outlier score')

print("\n")
print("Printing 20 lowest KNN density points")
for j in range(0, 20):
    print("Index: " + str(i_KNN[j]) + "   -   Density: " + str(density[j]))

print("\n")
### K-nearest neigbor average relative density
# Compute the average relative density

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
density = 1./(D.sum(axis=1)/K)
avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]

# Plot k-neighbor estimate of outlier score (distances)
figure(5)
bar(range(20),avg_rel_density[:20])
title('KNN average relative density: Outlier score')

print("\n")
print("Printing 20 lowest KNN Average Relative density points")
for j in range(0, 20):
    print("Index: " + str(i_avg_rel[j]) + "   -   Density: " + str(avg_rel_density[j]))

print("\n")
print("Indices shared by two methods")
for j in range(0,20):
    for k in range(0,20):
        if (i_gaus[j] == i_KNN[k]):
            print("In both Gausian and KNN: " + str(i_gaus[j]))
            print(dataSetup.movie.iloc[j])
        elif (i_gaus[j] == i_avg_rel[k]):
            print("In both Gausian and Average Relative KNN: " + str(i_gaus[j]))
            print(dataSetup.movie.iloc[j])
        elif (i_KNN[j] == i_avg_rel[k]):
            print("In Both KNN and Average Relative KNN: " + str(i_KNN[j]))
            print(dataSetup.movie.iloc[j])
print("\n")

print("Indices shared by all 3 methods")
for j in range(0,20):
    for k in range(0,20):
        for h in range(0,20):
            if(i_gaus[j] == i_KNN[k] == i_avg_rel[h]):
                print("Index: " + str(i_gaus[j]))
print("\n")