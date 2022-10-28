import tensorflow as tf
from keras import layers, datasets, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


def get_dataset(data_params: dict):
    set = data_params["set"]
    n = data_params["n"]
    lenght = 10000
    if data_params["dataset"] == "cifar-10":
        (X_pretest, y_pretest), (X_test, y_test) = datasets.cifar10.load_data()
    X_trigger, y_trigger = X_pretest[0:lenght], y_pretest[0:lenght]
    X_train, y_train = X_pretest[lenght:], y_pretest[lenght:]
    # renvoyer n images aléatoire parmi le "set" entré en argument
    if set == "train":
        X = X_train
        y = y_train
    elif set == "test":
        X = X_test
        y = y_test
    else:
        X = X_trigger
        y = y_trigger
    X = list(X)
    index = np.random.randint(len(X), size=n)
    X_a = []
    y_a = []
    for i in index:
        X_a.append(X[i])
        y_a.append(y[i])
    # X_set_sample.shape =( n, 32, 32, 3) , y_set_sample = (n,1)
    return(np.array(X_a), np.array(y_a))


if __name__ == "__main__":
    # test when coding this module
    X, y = get_dataset({"set": "train", "n": 10, "dataset": "cifar-10"})
    plt.imshow(X[0])
    plt.show()
    print(y[0])
