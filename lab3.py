import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing

import sklearn.metrics, math
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt;

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

acc_sq = []

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

k = 2
num_val_samples = len(train_data) // k
num_epochs = 1000
all_scores = []

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def write_graphics(H, y_pred):
    regressor = LinearRegression()
    regressor.fit(test_targets.reshape(-1, 1), y_pred)
    y_fit = regressor.predict(y_pred)

    reg_intercept = round(regressor.intercept_[0], 4)
    reg_coef = round(regressor.coef_.flatten()[0], 4)
    reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

    print("\n")
    print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(test_targets, y_pred))
    print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(test_targets, y_pred))

    regressor = LinearRegression()
    regressor.fit(test_targets.reshape(-1,1), y_pred)
    y_fit = regressor.predict(y_pred)

    reg_intercept = round(regressor.intercept_[0],4)
    reg_coef = round(regressor.coef_.flatten()[0],4)
    reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

  #  plt.scatter(test_targets, y_pred, color='blue', label= 'data')
  #  plt.plot(y_pred, y_fit, color='red', linewidth=2, label = 'Linear regression\n'+reg_label)
  #  plt.title('Linear Regression')
  #  plt.legend()
  #  plt.xlabel('observed')
  #  plt.ylabel('predicted')
  #  plt.show()

    plt.plot(H.history['mean_absolute_error'], color='blue', label='mae')
    plt.plot(H.history['val_mean_absolute_error'], color='red', label='val mae')
    plt.title('MAE')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('mean absolute error')
    plt.show()


for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)

    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    H = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(val_data, val_targets))
    pred = model.predict(test_data)
    write_graphics(H, pred)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(np.mean(all_scores))