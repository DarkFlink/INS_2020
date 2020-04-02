import numpy as np
import csv

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

import sklearn.metrics, math
from sklearn.linear_model import LinearRegression

dataset_size = 200
dataset_dim = 6
encode_dim = 2

def writeToCSV(path, dataset):
    with open(path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                        quoting=csv.QUOTE_MINIMAL)
        for item in dataset:
            spamwriter.writerow(item)

def genDataset(size):
    dataset = []
    dataset_y = []
    for i in range(size):
        X = np.random.normal(-5, 10)
        e = np.random.normal(0, 0.3)
        x = []
        y = []
        x.append(np.round((-X)**3 + e, decimals=3))
        x.append(np.round(np.log(np.fabs(X)) + e, decimals=3))
        x.append(np.round(np.sin(3*X) + e, decimals=3))
        x.append(np.round(np.exp(X) + e, decimals=3))
        x.append(np.round(X + 4 + e,decimals=3))
        x.append(np.round(-X + np.sqrt(np.fabs(X)) + e, decimals=3))
        y.append(np.round(X + e, decimals=3))
        dataset.append(x)
        dataset_y.append(y)
    return np.round(np.array(dataset),decimals=3), np.array(dataset_y)

def create_deep_dense_ae():
    encoding_dim = 3

    # Энкодер
    input_img = Input(shape=(dataset_dim))
    flat_img = Flatten()(input_img)
    encoded = Dense(30, activation='relu')(flat_img)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    # Декодер
    input_encoded = Input(shape=(encoding_dim,))
    flat_decoded = Dense(30, activation='relu')(input_encoded)
    flat_decoded = Dense(dataset_dim, activation='linear')(flat_decoded)

    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, flat_decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# data generation
x_train, y_train = genDataset(dataset_size)
x_test, y_test = genDataset(int(dataset_size/4))

# data normalization
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_train -= y_mean
y_train /= y_std
y_test -= y_mean
y_test /= y_std

print(y_train)
print(y_test)


# create (en/de)coder models
encoder, decoder, autoencoder = create_deep_dense_ae()

# autoencoder fitting
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()
H = autoencoder.fit(x_train, x_train,
                epochs=40,
                batch_size=1,
                shuffle=True,
                verbose=1,
                validation_data=(x_test, x_test))

# test (en/de)coder models
encoded_data = encoder.predict(x_test)
decoded_data = decoder.predict(encoded_data)

# regressor fitting
regressor = build_model()
H_r = regressor.fit(x_train, y_train,
                    epochs=50,
                    batch_size=2,
                    verbose=1,
                    validation_data=(x_test, y_test))

loss = H_r.history['loss']
v_loss = H_r.history['val_loss']
x = range(1, 50+1)

# training data plot
plt.plot(x, loss, 'b', label='train')
plt.plot(x, v_loss, 'r', label='validation')
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

# regressor preditions
regr_predicted_data = regressor.predict(x_test)

# regressor plot
#regressor_lin = LinearRegression()
#regressor_lin.fit(y_test.reshape(-1, 1), regr_predicted_data)
#y_fit = regressor_lin.predict(regr_predicted_data)
#
#reg_intercept = round(regressor_lin.intercept_[0], 4)
#reg_coef = round(regressor_lin.coef_.flatten()[0], 4)
#reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

#print("\n")
#print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test, regr_predicted_data))
#print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test, regr_predicted_data))

#plt.scatter(y_test, regr_predicted_data, color='blue', label= 'data')
#plt.plot(regr_predicted_data, y_fit, color='red', linewidth=2, label = 'Linear regression\n'+reg_label)
#plt.title('Linear Regression')
#plt.legend()
#plt.xlabel('observed')
#plt.ylabel('predicted')
#plt.show()

# seeing first predictions
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

print(x_test[0])
print(decoded_data[0])

print(y_train[0: 5])
print(regr_predicted_data[0: 5])

# write to csv
writeToCSV('./resource/x_train.csv',x_train)
writeToCSV('./resource/y_train.csv',y_train)
writeToCSV('./resource/x_test.csv',x_test)
writeToCSV('./resource/y_test.csv',y_test)
writeToCSV('./resource/encoded.csv',encoded_data)
writeToCSV('./resource/decoded.csv',decoded_data)
writeToCSV('./resource/regression_predicted.csv',regr_predicted_data)

#save models
decoder.save('decoder.h5')
encoder.save('encoder.h5')
regressor.save('regressor.h5')



