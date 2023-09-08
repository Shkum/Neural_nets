import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Стандартизация входных данных
x_train = x_train / 255
x_tet = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


size_val = 10_000                       # размер выборки валидации

x_val_split = x_train[:size_val]        # выделяем первые наблюдения из обучающей выборки
y_val_split = y_train_cat[:size_val]    # в выборку валидации

x_train_split = x_train[size_val:]      # выделяем последующие наблюдения для обучающей выборки
y_train_split = y_train_cat[size_val:]

# отображение первых 25 изображений из обучающе выборки
plt.figure(figsize=(10, 5))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary())

### myAdam = keras.optimizers.Adam(learning_rate=0.1)

mySgd = keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=True)

model.compile(optimizer=mySgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

###################################################################
from sklearn.model_selection import train_test_split  # random divide train and val sets
####################################################################

x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train_cat, test_size=0.2)

history = model.fit(x_train_split, y_train_split, batch_size=32, epochs=10, validation_data=(x_val_split, y_val_split))

n = 151
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f'Распознана цифра: {np.argmax(res)}')

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
print(pred.shape)
print(pred[:20])
print(y_test[:20])

mask = pred == y_test
print(mask[:10])

x_false = x_test[~mask]
p_false = pred[~mask]

print(x_false.shape)

for i in range(5):
    print('Значение сети: ' + str(p_false[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()
