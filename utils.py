import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_decisions_boundary(clf, X, Y):
    h = 0.2
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=.8)
    plt.scatter(X[:,0], X[:,1], c=Y)
