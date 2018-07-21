"""
ML for Practical Hackers

This is just a simple tutorial on k-NN algorithm on a low dimension case.
"""
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, cross_val_score

samples = 500
#Create simple non-linear dataset
X,Y = make_moons(n_samples=samples, noise=0.25)

#Plot dataset with a scatter
plt.scatter(X[:,0],X[:,1], c=Y)
plt.title('Twin moon dataset')
plt.show()

#Divide train and test, forget about the latter for a while
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

test_errors = [] 
errors = []
stds = []
cv_numb = 10
upper_bound_k = len(X_train) - (len(X_train) // cv_numb)-1
print('K from 1 to {}'.format(upper_bound_k))
K = range(0, upper_bound_k,10)
for k in K:
    #Create knn classifier and fit with
    knn = kNN(n_neighbors=k+1)

    #Perform cross validation with cv_numb divisions
    results = cross_val_score(knn, X_train, y_train, cv=cv_numb)

    #Mean error and std for this step of cross-validation
    mean_error = np.array(results).sum() / cv_numb
    std = np.array(results).std()

    errors.append(1 - mean_error)
    stds.append(std)

    #How this k performs on future data? Retrain with that k and whole set and score test
    tstknn = kNN(n_neighbors = k+1).fit(X_train, y_train)
    y_pred = tstknn.predict(X_test)
    test_errors.append( 1 - np.sum(y_pred == y_test) / len(y_pred) )

#Plot everything!
plt.title('kNN - Error for range of K')
plt.errorbar(K,errors,yerr=stds, label='Mean Validation')
plt.plot(K, test_errors, label='Test')
plt.xlabel('K')
plt.ylabel('Error')
plt.legend()
plt.show()

