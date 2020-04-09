import tensorflow.keras as keras
import re

from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout, Dense, Embedding, GRU
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

from tensorflow.keras import datasets

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


custom_x = ["it is very boring, bad film",
            "fantastic, the best film ever",
            "i think this film is the worst film i've seen, nothing interesting, junk",
            "fantastic film, wonderful casting, good job, creators",
            "beautiful picture, good scenario, it's amazing"]
custom_y = [0., 1., 0., 1., 1.]


class ReviewsClassifier(keras.Model):
    def __init__(self, max_features=5000, max_len=1000):
        super(ReviewsClassifier, self).__init__()

        self.weight_init = initializers.normal
        self.max_features = max_features
        self.max_len = max_len

        self.features = Sequential([
            Embedding(input_dim=self.max_features, output_dim=128, input_length=self.max_len),
            Dropout(0.4),

            GRU(input_dim=128, units=64, use_bias=True, kernel_regularizer=regularizers.l2(0.001)),

            Dense(input_dim=64, units=32, activation='relu'),
            Dropout(0.3),

            Dense(units=1, activation='sigmoid')
        ])

    def call(self, inputs):
        x = self.features(inputs)
        return x


# funcs for generating data for net from reviews
def encodeUserInput(string, dic):
    str_list = re.split('; |, |\*|\n| ', string)
    for i in range(len(str_list)):
        idx = dic.get(str_list[i])
        str_list[i] = idx
    return str_list


def generateReviews(reviews,dic):
    for i in range(len(reviews)):
        reviews[i] = encodeUserInput(reviews[i], dic)
    return reviews


# constants
batch_size = 200
epochs = 2
max_len = 250
max_features = 10000

# load data
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(seed=911, num_words=max_features)

# custom data testing
index = datasets.imdb.get_word_index()
dic = dict(index)

custom_x = generateReviews(custom_x, dic)

# prepare data
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
custom_x = sequence.pad_sequences(custom_x, maxlen=max_len)

X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=123)

print(x_train.shape)

y_test = np.asarray(y_test).astype("float32")
y_train = np.asarray(y_train).astype("float32")
custom_y = np.asarray(custom_y).astype("float32")

# init model
classifier = ReviewsClassifier(max_features=max_features,max_len=max_len)
optimizer = optimizers.Adam(lr=0.001)
loss = losses.BinaryCrossentropy()

classifier.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
H = classifier.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# testing
test_loss, test_acc = classifier.evaluate(x_test, y_test)
print('test_acc:', test_acc)

# custom testing
custom_loss, custom_acc = classifier.evaluate(custom_x, custom_y)
print('custom_acc:', custom_acc)
preds = classifier.predict(custom_x)

# plot
plt.figure(3,figsize=(8,5))
plt.title("Custom dataset predications")
plt.plot(custom_y, 'r', marker='v', label='truth')
plt.plot(preds, 'b', marker='x', label='pred')
plt.legend()
plt.show()
plt.clf()

plt.figure(1,figsize=(8,5))
plt.title("Training and test accuracy")
plt.plot(H.history['acc'], 'r', label='train')
plt.plot(H.history['val_acc'], 'b', label='test')
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

