import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from pwn import process
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split, cross_val_score


x = np.array([1,1,1,1])
y = np.array( [ord(a) / 256.0 for a in 'flag'] )

X,Y = np.matrix([x]), np.matrix([y])

model = keras.Sequential(
    [
        keras.layers.Dense(4, activation=keras.activations.sigmoid, input_shape=(4,), kernel_initializer=keras.initializers.Zeros())
    ]
)

model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(lr=1e-2), metrics=['accuracy'])
model.fit(X, Y, epochs=300)
y_pred = model.predict(X)
print(y_pred, Y)
response = ''.join( [chr(int(256*a)) for a in y_pred.ravel()] )
print('Response is ... ', response)

layer = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]
print(layer,biases)

print("IT'S MORE EASY THAN TATH!")

bias = [ np.log(i /(1-i)) for i in y ]
print(bias)
conn = process('Challenge/Pilgrim/NNawkward_pilgrim')
conn.sendline('1')
conn.sendline('3')
conn.sendline()
for _ in range(16):
        conn.sendline('0')
conn.sendline('4')
conn.sendline()
for i in range(4):
    conn.sendline('0')
for i in range(4):
    conn.sendline('%.14f' % bias[i])
conn.sendline('2')
conn.interactive()

#[-0.41197978912935806, -0.31508104663989539, -0.49419322371684876, -0.39570893316279959]