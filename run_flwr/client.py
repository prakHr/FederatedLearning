import warnings
import flwr as fl
from flwr.server.strategy import FedAvg

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils
import seaborn as sns
import matplotlib.pyplot as plt
def getDist(y):
    ax = sns.countplot(y)
    ax.set(title="Count of data classes")
    plt.show()
if __name__ == "__main__":
    # Load MNIST dataset from https://www.openml.org/d/554
    (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(2)
    (X_train, y_train) = utils.partition(X_train, y_train, 2)[partition_id]
    getDist(y_train)

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    def client_fn(cid: str):
        # Return a standard Flower client
        return MnistClient()
    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:5040", client=MnistClient())
    # Launch the simulation
    # hist = fl.simulation.start_simulation(
    #     client_fn=client_fn, # A function to run a _virtual_ client when required
    #     num_clients=50, # Total number of clients available
    #     config=fl.server.ServerConfig(num_rounds=3), # Specify number of FL rounds
    #     strategy=FedAvg() # A Flower strategy
    # )