import dataset
import models.classifiers.classifier_thomas as classi
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
    'nb_epochs': 4,
    'learning_rate': 1e-3,
    'type': 'convo',  # or 'dense'
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

trigger_params = trigger_params = dict
{
    "n": 120,
    "batch_size": 32,
    "nb_epochs": 3,
    "nb_app_epoch": 3
}

model_params = {
    "saved": None,
    "to save": "model2-thomas",
    "classifier": "classifier1",
    "hyperparams": hyperparams,
    "wm": trigger_params,
}

data_params = {
    "dataset": "cifar-10",
    "set": "train",
    "n": 200,
}


mod, param = classi.get_model(
    model_params=model_params, data_params=data_params)

# classi.save(mod, model_params=model_params)
