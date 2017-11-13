# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:18:18 2017

@author: Meowasaurus
"""


import numpy as np
import pandas as pd

import dataSetup

print('\n')

df = dataSetup.numbersData
df_imdb=dataSetup.numbersData['imdb_score']

q50 = df_imdb.quantile(.5)

#print(df_imdb)

bins = [0,q50]
group_names = ['bad','good']

X = df.drop('imdb_score',axis=1).values

y1 = df_imdb.values
y = np.digitize(y1,bins)-1





N, M = X.shape

attributeNames = df.drop('imdb_score',axis=1).columns.values

classNames = group_names

C = len(classNames)


