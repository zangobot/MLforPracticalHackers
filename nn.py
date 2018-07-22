import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split, cross_val_score

X,Y = make_moons(n_samples=600, noise=0.15)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()
model = keras.Sequential(
    [
        keras.layers.Dense(2, activation=keras.activations.sigmoid, input_shape=X[0].shape, kernel_initializer=keras.initializers.RandomNormal()),
        keras.layers.Dense(2, activation=keras.activations.sigmoid, kernel_initializer=keras.initializers.RandomNormal()),
        keras.layers.Dense(1, activation=keras.activations.sigmoid, kernel_initializer=keras.initializers.RandomNormal())
    ]
)
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=20)
y_pred = model.predict_classes(X_test)
print('ACCURACY: {}'.format( 100 * np.sum(y_pred.ravel() == y_test) / len(y_test)))
