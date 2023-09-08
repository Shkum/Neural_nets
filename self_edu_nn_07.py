import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense


c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

model = keras.Sequential()
model.add(Dense(units=1, input_shape=(1,), activation='linear'))
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.1))

history = model.fit(c, f, epochs=2000, verbose=0)

print(model.predict(np.array([100])))

print(model.get_weights())

plt.plot(history.history['loss'], 'g', label='Loss')
# plt.plot(history.history['accuracy'], 'b')
plt.grid(True)
plt.show()

