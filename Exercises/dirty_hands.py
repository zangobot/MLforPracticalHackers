import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import * #take solution of previou exercise...

X,Y = 0,0# use solution of previous

#DIVIDE TEST AND TRAIN!
X_tr, X_ts, y_tr, y_ts = ...

grid_parameters = {
    'C'         : ...,                      # ... space for regularization parameter,
    'coef0'     : ...,                      # ... space for poly degree
    'gamma'     : ...,                      # ... space for std in gaussian or constant multiplier in poly
    'kernel'    : ['lin', 'rbf', 'poly']    # ... kernels = linear, gaussian and polynomail
}

clf = GridSearchCV(SVC(), param_grid=grid_parameters, cv=5)

#Be careful of what you fit...
clf.fit( ... )

print(clf.best_params_)
best = clf.best_estimator_

#Be careful of what you predict...
y_pred = best.predict( ... )

#Score error!
error = np.sum( y_pred != ... ) / len(y_pred) * 100
print('Error of test is: {}'.format(error))
