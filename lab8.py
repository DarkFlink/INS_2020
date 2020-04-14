import tensorflow.keras as keras
import sys

from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout, Dense, GRU, Conv1D, MaxPool1D
from tensorflow.keras.models import Sequential

from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np

# constants
filename = "alice_in_the_wonderland.txt"
batch_size = 128
epochs = 10


def genText(dataX, int_to_char, model, pattern, gen_len=1000):
	# pick a random seed
	print("Seed:")
	print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
	# generate characters
	for i in range(gen_len):
		x = np.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0)
		index = np.argmax(prediction)
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in pattern]
		sys.stdout.write(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	print("\nDone.")


class TextGenerator(keras.Model):
    def __init__(self, patterns=10, input_shape=(1,1)):
        super(TextGenerator, self).__init__()

        self.weight_init = initializers.normal
        self.patterns = patterns

        self.features = Sequential([

            Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            MaxPool1D(pool_size=2),
            Dropout(0.1),

			GRU(input_shape=input_shape, units=256),
            Dropout(0.1),

            Dense(units=self.patterns, activation='softmax')
        ])

    def call(self, inputs):
        x = self.features(inputs)
        return x


# callback implementation
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, dataX, model, int_to_char, interval=2, epochs=epochs):
        super(CustomCallback, self).__init__()
        self.dataX = dataX
        self.epochs = epochs
        self.model = model
        self.int_to_char = int_to_char
        self.push_interval = interval
        start = np.random.randint(0, len(self.dataX) - 1)
        self.pattern = self.dataX[start]


    def on_train_begin(self, logs={}):
        self.sample = []

    def on_epoch_begin(self, epoch, logs={}):
        if (epoch + 1) % self.push_interval == 0 or epoch == 0 or (epoch + 1) == self.epochs:
            genText(gen_len=200, int_to_char=self.int_to_char, model=self.model, dataX=self.dataX, pattern=self.pattern)


# load data
raw_text = open(filename).read()
raw_text = raw_text.lower()

# prepare data
unique_chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(unique_chars))
int_to_char = dict((i, c) for i, c in enumerate(unique_chars))

n_chars = len(raw_text)
n_vocab = len(unique_chars)

# convert chars to it's integer keys
dataX = []
dataY = []
seq_length = 100
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape && normalize
X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = to_categorical(dataY)

# init model
model = TextGenerator(input_shape=(X.shape[1], X.shape[2]), patterns=y.shape[1])
optimizer = optimizers.Adam(lr=0.001)
loss = losses.CategoricalCrossentropy()

callback = CustomCallback(dataX=dataX, int_to_char=int_to_char, model=model)
model.compile(optimizer=optimizer, loss=loss)
model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[callback])

start = np.random.randint(0, len(dataX) - 1)
pattern = dataX[start]
genText(dataX=dataX, model=model, int_to_char=int_to_char, gen_len=1000,pattern=pattern)