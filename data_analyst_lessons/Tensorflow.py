import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

print('trainset:', X_train.shape)
print('testset:', X_test.shape)


X_train = X_train / 255
X_test = X_test / 255

fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(20, 4))
for i in range(10):
    ax[i].imshow(X_train[i], cmap='gray')

plt.tight_layout()
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])