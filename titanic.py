"""
ML for Practical Hackers

THE TITANIC CHALLENGE

Have fun creating a model for predict if a person would have or not survived the Titanic catastrophe.
There will be a final ranking, the one with the highest score will gain the flag.
"""

import numpy as np
import sklearn
import matplotlib.pyplot as plt
from utils import read_titanic_dataset, read_titanic_test_labels
from utils import plot_decisions_boundary

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, train_test_split

X_train, y_train = read_titanic_dataset('Challenge/Titanic/train.csv')

X_f, Y_f = read_titanic_dataset('Challenge/Titanic/test.csv', False), read_titanic_test_labels()


