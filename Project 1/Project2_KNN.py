# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:34:11 2017

@author: Meowasaurus
"""
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show

from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import Project2_Classification as setup
import numpy as np

# Maximum number of neighbors
L=40

K = 100
N = setup.N
X = setup.X
y = setup.y


CV = cross_validation.KFold(N,K,shuffle=True)
errors = np.zeros((K,L))
i=0
for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}'.format(i+1,K))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
        

    i+=1
# Plot the classification error rate
figure()
print(errors[1])
plot(sum(errors,0)/K)

best = 100
bestIndex = 0
sumErrors = (sum(errors,0)/K) 
for i in range(L):
    if(best > sumErrors[i]):
        best = sumErrors[i];
        bestIndex = i
 
xlabel('Number of neighbors')
ylabel('Classification error rate')
show()
