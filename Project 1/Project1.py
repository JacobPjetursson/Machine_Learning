#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:20:42 2017

@author: Jacob
"""

from numpy import genfromtxt

movieData = genfromtxt('movie_metadata.csv', delimiter=',',dtype=None, usecols=(1,2,3,8,9,11,12,16,18,19,20,21,22,23,25,26))
#with open('movie_metadata.csv', newline='') as csvfile:
 #   movieData = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
for row in movieData:
    print(row)
        
