import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Function to load the dataset
def load_data(filename):
    """
    Loads the data from a given filename.

    :param filename: Path to the file.
    :return: Loaded dataset.
    """
    return pd.read_csv(filename).iloc[:, 1].values.astype("float32").reshape(-1, 1)

# Function to scale the dataset
def scale_data(data):
    """
    Scales the data using MinMax scaling.

    :param data: Data to be scaled.
    :return: Scaler object and scaled data.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler, scaler.fit_transform(data)

# Function to split the dataset into train and test
def split_data(dataset, train_size_ratio=0.85):
    """
    Splits the data into training and testing datasets.

    :param dataset: Data to be split.
    :param train_size_ratio: Ratio for the train dataset.
    :return: Training and testing datasets.
    """
    train_size = int(len(dataset) * train_size_ratio)
    return dataset[0:train_size], dataset[train_size:]

# Function to prepare the dataset
def prepare_data(data, time_stemp=10):
    """
    Prepares the data for the RNN.

    :param data: Data to be prepared.
    :param time_stemp: Time step for splitting the data.
    :return: Prepared X and Y datasets.
    """
    X, Y = [], []
    for i in range(len(data) - time_stemp - 1):
        X.append(data[i:(i + time_stemp), 0])
        Y.append(data[i + time_stemp, 0])
    return np.array(X), np.array(Y)
