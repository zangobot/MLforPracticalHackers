"""
ML for Practical Hackers

This is just a simple tutorial on SVM algorithm.
"""
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV

X, Y = make_moons(n_samples=600, noise=1.2)
plt.scatter(X[:,0], X[:,1], c=Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

grid = {
    'C'         : np.logspace(-2,2,20),
    'kernel'    : ['linear', 'rbf'],
    'gamma'     : np.logspace(-1,1,10),

}
clf = GridSearchCV(SVC(), param_grid=grid, cv=10)
clf.fit(X_train, y_train)
clf.best_estimator_