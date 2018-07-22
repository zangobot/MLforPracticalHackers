"""
ML for Practical Hackers

This is just a simple tutorial on SVM algorithm.
"""
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from utils import plot_decisions_boundary

from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, train_test_split

X, Y = make_moons(n_samples=600, noise=0.25)
plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

grid_linear = {
    'C'         : np.logspace(-2,2,10),
    'kernel'    : ['linear']

}
clf_linear = GridSearchCV(SVC(), param_grid=grid_linear, cv=10)
clf_linear.fit(X_train, y_train)
linear_error = 100 * np.sum(clf_linear.best_estimator_.predict(X_test) != y_test) / len(y_test)


grid_gauss = {
    'C'         : np.logspace(-2,2,10),
    'kernel'    : ['rbf'],
    'gamma'     : np.logspace(-1,1,10),

}
clf_gauss = GridSearchCV(SVC(), param_grid=grid_gauss, cv=10)
clf_gauss.fit(X_train, y_train)
gauss_error = 100 * np.sum(clf_gauss.best_estimator_.predict(X_test) != y_test) / len(y_test)

print('Linear SVM, Test Error = {0:.2f}'.format(linear_error))
print('Guassian Kernel SVM, Test Error = {0:.2f}'.format(gauss_error))
plt.subplot(121)
plt.title('Linear SVM decision function')
plot_decisions_boundary(clf_linear.best_estimator_, X_test, y_test)
plt.colorbar()
plt.subplot(122)
plt.title('Gaussian SVM decision function')
plot_decisions_boundary(clf_gauss.best_estimator_, X_test, y_test)
plt.colorbar()
plt.show()
