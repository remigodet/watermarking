from models.classifiers.classifier_thomas import get_model
from dataset import get_dataset
import tensorflow as tf
from keras import layers, datasets, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

hyperparams = {
    "train_ratio": 0.5,
    "val_ration": 0.3,
    "test_ration": 0.2,
    "batch_size": 32,
    'nb_epochs': 10,
    'learning_rate': 1e-3,
    'archi': 'dense',  # or 'convo'
    'kernel_size': (3, 3),
    'activation': 'relu',
    'nb_targets': 10,
    'nb_layers': 2,
    'add_pooling': True,
    'pooling_size': (2, 2),
    'nb_units': [32, 64],
    'optimizer': keras.optimizers.Adam,
    'loss': 'sparse_categorical_crossentropy',  # 'metrics' : ['accuracy'],
}

trigger_params  = dict
{
    "n": 120,
    "nb_app_epoch": 3
}

model_params = {
    "saved": None,
    "to save": None,
    "classifier": "classifier1",
    "hyperparams": hyperparams,
    "wm": None,
}

data_params = {
    "dataset": "cifar-10",
    "set": "train",
    "n": 40000,
    "seed": 42
}

X_test, y_test =get_dataset({"n":3000,"set": "test","dataset":"cifar-10", "seed": 42  })
model1=get_model(model_params, data_params, model=None)
print('Model 1 calculated')
model2=get_model(model_params, data_params, model=model1)
model2.evaluate(X_test, y_test)
print('Model 2 calculated')