import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

def plot_decisions_boundary(decision_function, X, Y, alpha=.8):
    h = 0.2
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=alpha)
    plt.scatter(X[:,0], X[:,1], c=Y)

def read_titanic_dataset(path, has_labels=True):
    train = pd.read_csv(path)

    #Deal with missing values!
    train['Age'].fillna(train['Age'].mean(), inplace=True)
    train['Fare'].fillna(train['Fare'].mean(), inplace=True)

    #Encode categorical string into numbers
    train = train.replace(to_replace={'Sex': {'male': 0, 'female':1}})
    column_to_remove = ['Cabin', 'Name', 'Embarked', 'PassengerId', 'Ticket']
    if has_labels:
        column_to_remove.append('Survived')
        labels = train['Survived']
    train = pd.get_dummies(train, columns=['Pclass'])
    #Remove not-so useful columns (except the ticket... we could infer something from it?) and Y (if present)
    train.drop(column_to_remove,axis=1, inplace=True)
    X = train.as_matrix() 
    result = X if not has_labels else (X, labels.as_matrix().ravel()) 
    return result

def read_titanic_test_labels():
    labels = pd.read_csv('Challenge/Titanic/gender_submission.csv')
    return labels.as_matrix()[:,1].ravel()