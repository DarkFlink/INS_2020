import numpy as np

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPool1D, Dense, Flatten, Conv1D
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import losses
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from var6 import gen_data

class ClassifierFigs(keras.Model):
    def __init__(self, num_classes=3):
        super(ClassifierFigs, self).__init__()

        self.initializer = initializers.normal
        self.num_classes = num_classes

        self.features = Sequential([
            Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', use_bias=True),
            MaxPool1D(pool_size=2),

            Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', use_bias=True),
            #Dropout(0.3),
            Conv1D(filters=128, kernel_size=5, strides=1, activation='relu'),
            MaxPool1D(pool_size=3),
        ])

        self.lin = Sequential([
            Flatten(),
            Dense(units=self.num_classes, activation='softmax'),
        ])

    def call(self, inputs):
        x = self.features(inputs)
        print(x.shape)
        x = self.lin(x)
        return x


epochs = 17
batch_size = 1


# callback implementation
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, y_train, interval=5, epochs=epochs):
        super(CustomCallback, self).__init__()
        self.y_train = y_train
        self.epochs = epochs
        self.push_interval = interval

    def on_train_begin(self, logs={}):
        self.sample = []

    def on_epoch_begin(self, epoch, logs={}):
        self.worst_batch_index = 0
        self.worst_batch_acc = 1.0
        self.worst_batch_loss = 0
        self.worst_batch_class = 0

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1) % self.push_interval == 0 or epoch == 0 or (epoch+1) == self.epochs:
            epoch_data = [(epoch+1)]
            epoch_data.append(self.worst_batch_index)
            epoch_data.append(self.worst_batch_acc)
            epoch_data.append(self.worst_batch_loss)
            epoch_data.append(self.worst_batch_class)
            self.sample.append(epoch_data)

    def on_batch_end(self, batch, logs={}):
        acc = logs.get('acc')
        if self.worst_batch_acc > float(acc):
            self.worst_batch_acc = acc
            self.worst_batch_index = batch
            self.worst_batch_loss = logs.get('loss')
            self.worst_batch_class = self.y_train[batch]


# preparing data
X, Y = gen_data()

X = np.asarray(X)
Y = np.asarray(Y).flatten()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

print(y_test[0:10])

y_train_non_categ = y_train

le = preprocessing.LabelEncoder()
le.fit(Y)

y_test = np.asarray(le.transform(y_test))
y_train = np.asarray(le.transform(y_train))

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

x_train = np.asarray(x_train) / 255
x_test = np.asarray(x_test) / 255

# model prepare
classifier = ClassifierFigs()
optimizer = optimizers.Adam(lr=0.0001)
criterion = losses.CategoricalCrossentropy()

history = CustomCallback(y_train_non_categ)

# model fit
classifier.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])
H = classifier.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[history])

# testing
test_loss, test_acc = classifier.evaluate(x_test, y_test)
print('test_acc:', test_acc)

# callback history
table = np.asarray(history.sample)
df = pd.DataFrame(table, columns= ['epoch', 'batch_index', 'acc', 'loss', 'truth_class'])
df.to_csv('table.csv', index=False)

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