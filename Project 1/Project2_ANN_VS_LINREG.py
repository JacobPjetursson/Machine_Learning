#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:57:18 2017

@author: Jacob
"""
from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
import numpy as np
from scipy import stats
import Project2_LinearReg
import Project2_ANN

errors_linReg = Project2_LinearReg.Error_test
errors_ANN = Project2_ANN.errors
K = Project2_ANN.K


# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05. 

"""
K = 5
Error_linearReg = np.empty((K,1))
Error_ANN = np.empty((K,1))

for i in range(K):
    Error_linearReg[i] = 0.46
    Error_ANN[i] =       0.4
"""
z = (errors_linReg-errors_ANN)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Regression types are not significantly different')        
else:
    print('Regression types are significantly different.')
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.bmat('errors_linReg, errors_ANN'))
xlabel('Linear Regression   vs.   ANN')
ylabel('Cross-validation mean square error [%]')

show()