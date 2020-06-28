from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import IsolationForest
from time import time
import numpy as np

NSL_KDD_PATH = "/Users/isaac/temp/NSL-KDD-Dataset/"
TRAIN_FILE = 'KDDTrain+.txt'
TEST_FILE = 'KDDTest+.txt'


def load_nsl_kdd_train_set():
    with open(NSL_KDD_PATH + TRAIN_FILE) as input_file:
        lines = input_file.readlines()
        lines = list(map(lambda x: x.split(','), lines))
        data = list(map(lambda x: x[:-2], lines))
        target = list(map(lambda x: x[-2], lines))
        return data, target


def load_nsl_kdd_test_set():
    with open(NSL_KDD_PATH + TEST_FILE) as input_file:
        lines = input_file.readlines()
        lines = list(map(lambda x: x.split(','), lines))
        data = list(map(lambda x: x[:-2], lines))
        target = list(map(lambda x: x[-2], lines))
        return data, target


def get_encoder(data):
    enc = OrdinalEncoder()
    sub_data = list(map(lambda x: x[1:4], data))
    enc.fit(sub_data)
    return enc


def encode_nsl(enc, data):
    sub_data = list(map(lambda x: x[1:4], data))
    sub_data = enc.transform(sub_data)
    data = list(map(lambda item: list(item[0][0]) + list(item[1]) + list(item[0][4:]), zip(data, sub_data)))
    data = list(map(lambda x: [float(i) for i in x], data))
    for d in data:
        if len(d) != 41:
            print(len(d), d)
    return data


if __name__ == '__main__':
    data, target = load_nsl_kdd_train_set()
    encoder = get_encoder(data)
    X_train = encode_nsl(encoder, data)
    print(X_train)

    X_test, y_test = load_nsl_kdd_test_set()
    X_test = encode_nsl(encoder, X_test)

    random_state = 1
    print('--- Fitting the IsolationForest estimator...')
    model = IsolationForest(n_jobs=-1, random_state=random_state, verbose=True)
    tstart = time()
    model.fit(X_train)
    fit_time = time() - tstart
    tstart = time()

    scoring = - model.decision_function(X_test)  # the lower, the more abnormal
    print(scoring)
