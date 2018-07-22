"""
ML for Practical Hackers

This is just a simple tutorial on Random Forest algorithm.
"""
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from utils import plot_decisions_boundary

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, train_test_split

X,Y = make_moons(n_samples=600, noise=0.25)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

grid_gauss = {
    'C'         : np.logspace(-2,2,10),
    'kernel'    : ['rbf'],
    'gamma'     : np.logspace(-1,1,10),
}
clf_gauss = GridSearchCV(SVC(), param_grid=grid_gauss, cv=10)
clf_gauss.fit(X_train, y_train)
gauss_error = 100 * np.sum(clf_gauss.best_estimator_.predict(X_test) != y_test) / len(y_test)

rf = RandomForestClassifier(n_estimators=15)
rf.fit(X_train, y_train)
rf_error = 100 * np.sum(rf.predict(X_test) != y_test) / len(y_test)

print('Gaussian SVM, Test Error = {0:.2f}'.format(gauss_error))
print('Random Forest, Test Error = {0:.2f}'.format(rf_error))
plt.subplot(121)
plt.title('Gaussian SVM decision function')
plot_decisions_boundary(clf_gauss.best_estimator_.decision_function, X_test, y_test)
plt.colorbar()
plt.subplot(122)
plt.title('Random Forest decision function')
plot_decisions_boundary(rf.predict, X_test, y_test, alpha=0.5)
plt.colorbar()
plt.show()
