import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from utils import plot_decisions_boundary
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler

X,Y = make_circles(n_samples=500, noise=0.05, factor=0.4) #make_moons(n_samples=800, noise=0.15)
# plt.scatter(X[:,0],X[:,1],c=Y)
# plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# scaler = MinMaxScaler(feature_range=(0,1))
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

model = keras.Sequential(
    [
        keras.layers.Dense(3, activation=keras.activations.sigmoid, input_shape=X[0].shape, kernel_initializer=keras.initializers.RandomNormal()),
        # keras.layers.Dense(2, activation=keras.activations.sigmoid, kernel_initializer=keras.initializers.RandomNormal()),
        keras.layers.Dense(1, activation=keras.activations.sigmoid, kernel_initializer=keras.initializers.RandomNormal())
    ]
)

# Higher batch size => more epochs to reach good performance (few GD steps)
# Small batch size => less epochs but not precise gradient calculation (noisy)
# This means that maybe it won't converge in reasonable time.
#
# TRY IT! Use batch of 1 (SGD), 100, 200, 500 (GD) and tune epoch as well

model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(lr=1e-2), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0, batch_size=10)
y_pred = model.predict_classes(X_test)
print('ACCURACY: {}'.format( 100 * np.sum(y_pred.ravel() == y_test) / len(y_test)))
plot_decisions_boundary(model.predict_classes, X_train, y_train, alpha=0.5, step=1e-1)
plt.show()