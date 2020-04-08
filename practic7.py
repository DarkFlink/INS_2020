import numpy as np

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import losses
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from var6_task7 import gen_sequence

def gen_data_from_sequence(seq_len = 1006, lookback = 10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i,i+lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback,len(seq))])
    return (past, future)


class RecurrentNet(keras.Model):
    def __init__(self):
        super(RecurrentNet, self).__init__()

        self.features = Sequential([
            GRU(input_dim=1, units=128, recurrent_activation='relu', return_sequences=True),
            Dropout(0.5),
            GRU(input_dim=128, units=32, activation='relu', return_sequences=False),
        ])

        self.fc = Sequential([
            Dense(input_dim=32, units=1)
        ])

    def call(self, inputs):
        x = self.features(inputs)
        x = self.fc(x)
        return x


# consts
dataset_size = 2000
epochs = 10
batch_size = 10

# preparing data
X, Y = gen_data_from_sequence(dataset_size)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.08, random_state=123)

print(x_train.shape)
print(x_test.shape)

# preparing model
model = RecurrentNet()
optimizer = optimizers.Adam(lr=0.0001)
criterion = losses.MeanSquaredError()

# model fit
model.compile(optimizer=optimizer, loss=criterion, metrics=['mean_absolute_error'])
H = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# testing
preds = model.predict(x_test)

# plots
plt.figure(1, figsize=(20,12))
plt.title("Predictions")
plt.plot(preds, 'b', label='pred')
plt.plot(y_test, 'r', label='truth')
plt.legend()
plt.show()
plt.clf()

plt.figure(2,figsize=(8,5))
plt.title("Training and test loss")
plt.plot(H.history['loss'], 'r', label='train')
plt.plot(H.history['val_loss'], 'b', label='test')
plt.legend()
plt.show()
plt.clf()