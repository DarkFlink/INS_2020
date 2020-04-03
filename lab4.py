import numpy as np

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras import losses

import matplotlib.pyplot as plt

from PIL import Image

input_dim = 28 * 28
learn_rate = 0.001
epochs = 6
batch_size = 100


class Classifier(keras.Model):

    def __init__(self, num_classes=10):
        super(Classifier, self).__init__(name='example')

        self.initializer = initializers.normal
        self.num_classes = num_classes

        self.features = Sequential([
            Dense(input_dim=28 * 28, units=28 * 14, activation='relu', kernel_initializer=self.initializer),
            Dense(input_dim=28 * 14, units=28 * 5, activation='sigmoid', kernel_initializer=self.initializer),
            Dense(input_dim=28 * 5, units=self.num_classes, activation='softmax')
        ])

    def call(self, inputs):
        return self.features(inputs)


# function uploading image
def uploadImage(path):
    image = Image.open(path)
    data = np.asarray(image)
    return data


# uploading custom picture
data = uploadImage('pic.png')
print("Uploading TEST: " + data.shape)

# load dataset
mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# normalize
x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# init model
model = Classifier()
optimizer = optimizers.Adam(lr=learn_rate)
test1_opt = optimizers.SGD(lr=learn_rate, momentum=0.3)
test2_opt = optimizers.Adagrad(learn_rate)
test3_opt = optimizers.RMSprop(lr=learn_rate, rho=0.8)
loss = losses.CategoricalCrossentropy()

model.compile(optimizer=test3_opt, loss=loss, metrics=['accuracy'])
H = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# testing
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_acc:', test_acc)

# plot
plt.figure(1,figsize=(8,5))
plt.title("Training and test accuracy")
plt.plot(H.history['acc'], 'r', label='train')
plt.plot(H.history['val_acc'], 'b', label='test')
plt.legend()
plt.show()
plt.clf()

plt.figure(1,figsize=(8,5))
plt.title("Training and test loss")
plt.plot(H.history['loss'], 'r', label='train')
plt.plot(H.history['val_loss'], 'b', label='test')
plt.legend()
plt.show()
plt.clf()
