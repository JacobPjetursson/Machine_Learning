#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:20:42 2017

@author: Jacob
"""

from numpy import genfromtxt

movieData = genfromtxt('movie_metadata.csv', delimiter=',')
#with open('movie_metadata.csv', newline='') as csvfile:
 #   movieData = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
for row in movieData:
    print(row)

        
