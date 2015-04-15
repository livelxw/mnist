import pandas as pd
import numpy as np
from sklearn.preprocessing import binarize
from sklearn.cross_validation import train_test_split

num_labels = 10


def get_split_data(test_size=0.33, threshold=75):
    X, y = get_prepared_xy(threshold)
    trX, teX, trY, teY = train_test_split(X, y, test_size=test_size)
    return trX, teX, trY, teY


def get_raw_split_data(test_size=0.33):
    X, y = get_normalize_xy()
    trX, teX, trY, teY = train_test_split(X, y, test_size=test_size)
    return trX, teX, trY, teY


def get_prepared_xy(threshold=75):
    df = pd.read_csv("../data/train.csv")
    X = binarize(df.values[:, 1:], threshold=threshold)
    y = df.values[:, 0]
    y = np.eye(num_labels)[y, :]
    return X, y


def get_normalize_xy():
    df = pd.read_csv("../data/train.csv")
    X = df.values[:, 1:] / 255.
    y = df.values[:, 0]
    y = np.eye(num_labels)[y, :]
    return X, y