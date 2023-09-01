from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# import openml
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    # n_classes = 10  # MNIST has 10 classes
    # n_features = 784  # Number of features in dataset
    n_classes = 3  # IRIS has 10 classes
    n_features = 4  # Number of features in dataset
    
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def load_mnist() -> Dataset:
    """Loads the MNIST dataset using OpenML.

    OpenML dataset link: https://www.openml.org/d/554
    """
    # mnist_openml = openml.datasets.get_dataset(554)
    # Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    # X = Xy[:, :-1]  # the last column contains labels
    # y = Xy[:, -1]
    # First 60000 samples consist of the train set
    # split_ratio = 0.8
    # splitted_ratio = int(150*split_ratio)
    # x_train, y_train = X[:splitted_ratio], y[:splitted_ratio]
    # x_test, y_test = X[splitted_ratio:], y[splitted_ratio:]
    path = r"C:\Users\gprak\OneDrive\Desktop\FederatedLearning\dataset\archive\Iris.csv"
    df = pd.read_csv(path)
    le = preprocessing.LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])
    
    # print(df.head(10))
    
    X=df.drop(columns=['Species']).to_numpy()
    Y=df['Species'].to_numpy()

    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=1234)

    return (x_train, y_train), (x_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    print(len(list(np.array_split(X, num_partitions))))
    print(list(np.array_split(X, num_partitions)))
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
