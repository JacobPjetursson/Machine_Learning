#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:20:42 2017

@author: Jacob
"""
import numpy as np
from matplotlib.pyplot import boxplot, figure, subplot, hist, xlabel, ylim, show
from numpy import genfromtxt

movieData = genfromtxt('movie_metadata.csv', delimiter=',',dtype=None, names=True, usecols=(8,9,12,22,25))
numbersData = genfromtxt('movie_metadata.csv', delimiter=',', dtype=None, names=True, usecols=(8,12,22,25))
attributeNamesNumbers = numbersData.dtype.names
attributeNames = movieData.dtype.names


M = len(attributeNamesNumbers)
#for i in attributeNamesNumbers:
 #       print(i)
        #hist(movieData["" + i])
for i in attributeNames:
        print(i)
        
figure(figsize=(8,7))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
#for i in range(M):
   # subplot(u,v,i+1)
   # hist(movieData[:,i], color=(0.2, 0.8-i*0.2, 0.4))
  #  xlabel(attributeNamesNumbers[i])