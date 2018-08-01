import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import * #take solution of previou exercise...

X,Y = 0,0# use solution of previous

#DIVIDE TEST AND TRAIN!
X_tr, X_ts, y_tr, y_ts = ...

errors = []

for i in range(100):
    clf = RandomForestClassifier(n_estimators=...)

    #Be careful of what you fit...
    clf.fit( ..., ... )

    #Be careful of what you predict...
    y_pred = clf.predict( ... )

    #Score error!
    error = np.sum( y_pred != ... ) / len(y_pred) * 100
    print('Error of test is: {}'.format(error))

    errors.append(error)

#plot errors!