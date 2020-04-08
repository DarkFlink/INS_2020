import tensorflow.keras as keras

from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Dense, Flatten
from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets

import matplotlib.pyplot as plt

class ConvClassifier(keras.Model):
    def __init__(self, classes_count = 10):
        super(ConvClassifier, self).__init__()

        self.features = Sequential([
            Conv2D(filters=32, kernel_size=(7, 7), strides=(1,1), activation='relu'),
            Dropout(0.3),
            MaxPool2D(pool_size=2),

            Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
            Dropout(0.3),
            Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
            MaxPool2D(pool_size=2),
        ])

        self.lin = Sequential([
            Flatten(),
            Dense(input_dim=6*6*4, units=classes_count, activation='softmax'),
        ])

    def call(self, inputs):
        x = self.features(inputs)
        x = self.lin(x)
        return x

# constatnts
batch_size = 100
epochs = 20

# load data
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

print(x_train.shape)

# normalize data
x_train = x_train / 255
x_test = x_test / 255

y_test = to_categorical(y_test, num_classes=10)
y_train = to_categorical(y_train, num_classes=10)

# init model
classifier = ConvClassifier()
optimizer = optimizers.Adam(lr=0.001)
loss = losses.CategoricalCrossentropy()

classifier.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
H = classifier.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# testing
test_loss, test_acc = classifier.evaluate(x_test, y_test)
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