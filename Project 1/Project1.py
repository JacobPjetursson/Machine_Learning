#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:20:42 2017

@author: Jacob
"""
import numpy as np
from matplotlib.pyplot import boxplot, figure, subplot, hist, xlabel, ylim, show
from numpy import genfromtxt

movieData = genfromtxt('movie_metadata.csv', delimiter=',',dtype=None, names=True, usecols=(1,2,3,8,9,11,12,16,18,19,20,21,22,23,25,26))

attributeNames = movieData.dtype.names

#M = len(attributeNames)
for i in attributeNames:
   if(i != "director_name" and i != "genres" and i != "movie_title" and i != "plot_keywords" and i != "language" and i != "country" and i != "content_rating" and i != "aspect_ratio"):
        print(i)
        hist(movieData[:,""+i])
#hist(movieData["gross"])
#hist(movieData["num_critic_for_reviews"])
#hist(movieData["duration"])
#hist(movieData["num_voted_users"])
#hist(movieData["num_user_for_reviews"])
#hist(movieData["title_year"])
#hist(movieData["imdb_score"])
#hist(movieData["aspect_ratio"])
show()
#figure(figsize=(8,7))
#u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
#budget = movieData["budget"]
#hist(budget)
#boxplot(budget)
"""

classLabels = movieData.col_values(4,1,151)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

# Extract vector y, convert to NumPy matrix and transpose
y = np.array([classDict[value] for value in classLabels])

# Preallocate memory, then extract data to matrix X
X = np.empty((150,4))
for i in range(4):
    X[:,i] = np.array(movieData.col_values(i,1,151)).T

# Compute values of N, M and C.
N = len(y)
C = len(classNames)


for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i], color=(0.2, 0.8-i*0.2, 0.4))
    xlabel(attributeNames[i])
    ylim(0,N/2)
    
show()

"""
