from typing import Dict, Optional, Tuple
from pathlib import Path
import flwr as fl

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
# Start Flower server

if __name__=="__main__":

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

    def get_evaluate_fn(model):
        """Return an evaluation function for server-side evaluation."""

        # Load data and model here to avoid the overhead of doing it in `evaluate` itself
        # (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

        
        # The `evaluate` function will be called after every round
        def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, mse = model.evaluate(x_test, y_test)
            return loss, {"mean_squared_logarithmic_error": mse}

        return evaluate


    def fit_config(server_round: int):
        """Return training configuration dict for each round.

        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        config = {
            "batch_size": 32,
            "local_epochs": 1 if server_round < 2 else 2,
        }
        return config


    def evaluate_config(server_round: int):
        """Return evaluation configuration dict for each round.

        Perform five local evaluation steps on each client (i.e., use five
        batches) during rounds one to three, then increase to ten local
        evaluation steps.
        """
        val_steps = 5 if server_round < 4 else 10
        return {"val_steps": val_steps}


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
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )
    fl.server.start_server(
        server_address="localhost:5040",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
