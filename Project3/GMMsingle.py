# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:00:26 2017

@author: Meowasaurus
"""

# exercise 11.1.1
from matplotlib.pyplot import figure, show
import numpy as np
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from sklearn.mixture import GaussianMixture
from scipy import stats
from sklearn.decomposition import PCA
import Classification as cs

import random

bins = cs.bins
M = cs.M
df = cs.df.sample(n=1000)
df_imdb = df['imdb_score']

X = df.drop('imdb_score',axis=1).values

y1 = df_imdb.values
y = np.digitize(y1,bins)-1

X = stats.zscore(X)

pca = PCA(n_components=2)
x_2d = pca.fit_transform(X)



# Number of clusters
K = 5
cov_type = 'full'       
# type of covariance, you can try out 'diag' as well
reps = 10
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(x_2d)
cls = gmm.predict(x_2d)    
# extract cluster labels
cds = gmm.means_
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
#if cov_type == 'diag':    
new_covs = np.zeros([K,M,M])    

count = 0    
for elem in covs:        

    temp_m = np.zeros([M,M])        
    for i in range(len(elem)):
        for j in range(len(elem)):
            temp_m[i][j] = elem[i][j]        
            
    new_covs[count] = temp_m        
    count += 1
        
covs = new_covs
print(cds)
# Plot results:
figure(figsize=(14,9))
clusterplot(x_2d, clusterid=cls, centroids=cds, y=y, covars=covs)
show()


## In case the number of features != 2, then a subset of features most be plotted instead.
#figure(figsize=(14,9))
#idx = [0,1] # feature index, choose two features to use as x and y axis in the plot
#clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
#show()

print('Ran Exercise 11.1.1')