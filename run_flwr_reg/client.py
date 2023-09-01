import os

import flwr as fl
import tensorflow as tf
import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
TRAIN_DATA_PATH = './california_housing_train.csv'
TEST_DATA_PATH = './california_housing_test.csv'
TARGET_NAME = 'median_house_value'

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
# model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
train_data = pd.read_csv(TRAIN_DATA_PATH)
test_data = pd.read_csv(TEST_DATA_PATH)
x_train, y_train = train_data.drop(TARGET_NAME, axis=1), train_data[TARGET_NAME]
x_test, y_test = test_data.drop(TARGET_NAME, axis=1), test_data[TARGET_NAME]
def scale_datasets(x_train, x_test):

  """
  Standard Scale test and train data
  Z - Score normalization
  """
  standard_scaler = StandardScaler()
  x_train_scaled = pd.DataFrame(
      standard_scaler.fit_transform(x_train),
      columns=x_train.columns
  )
  x_test_scaled = pd.DataFrame(
      standard_scaler.transform(x_test),
      columns = x_test.columns
  )
  return x_train_scaled, x_test_scaled
x_train_scaled, x_test_scaled = scale_datasets(x_train, x_test)
x_train,x_test = x_train_scaled.to_numpy(),x_test_scaled.to_numpy()
y_train,y_test = y_train.to_numpy(),y_test.to_numpy()
# print(y_train.shape)
# print(x_train.shape)
hidden_units1 = 160
hidden_units2 = 480
hidden_units3 = 256
learning_rate = 0.01
# Creating model using the Sequential in tensorflow
def build_model_using_sequential(x_train):
    model = keras.Sequential()
    # For input layer
    model.add(Flatten(input_shape = x_train[0].shape))   # input layer
    model.add(Dense(hidden_units1, kernel_initializer='normal', activation='relu'))  
    model.add(Dropout(0.2))
    model.add(Dense(hidden_units2, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_units3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    
    return model
# build the model
model = build_model_using_sequential(x_train)

# loss function
msle = MeanSquaredLogarithmicError()
model.compile(
    loss=msle, 
    optimizer=Adam(learning_rate=learning_rate), 
    metrics=[msle]
)
# print(model.get_weights())
# print(model.evaluate(x_test, y_test))
# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # print(model.get_weights())
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        
        loss, mse = model.evaluate(x_test, y_test)

        return loss, len(x_test), {"mean_squared_logarithmic_error": mse}


# Start Flower client
fl.client.start_numpy_client(server_address="localhost:5040", client=CifarClient())
