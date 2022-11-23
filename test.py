import dataset
import models.classifiers.classifier_thomas as classi
import tensorflow as tf
from keras import layers, datasets, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
# creer
# ré entrainer simplement 2 fois
# wm sans pré entrainement, réentrainer wm
#

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
    "nb_app_epoch": 3
}

model_params = {
    "saved": "model1-thomas",
    "to save": None,
    "classifier": "classifier1",
    "hyperparams": None,
    "wm": None,
}

data_params = {
    "dataset": "cifar-10",
    "set": "train",
    "n": 200,
}


print('Test 1 Load uniquement  VALIDE')

# model_params = {
#     "saved": "model1-thomas",
#     "to save": None,
#     "classifier": "classifier1",
#     "hyperparams": None,
#     "wm": None,
# }

# data_params = {
#     "dataset": "cifar-10",
#     "set": "train",
#     "n": 200,
# }

# mod = classi.get_model(
#     model_params=model_params, data_params=data_params, model=None)

# print(mod)


print(
    'Test 2 {model = None, pas de load, entrainement simple, sans sauvegarde} Validé')

# hyperparams = {
#     "train_ratio": 0.5,
#     "val_ration": 0.3,
#     "test_ration": 0.2,
#     "batch_size": 32,
#     'nb_epochs': 3,
#     'learning_rate': 1e-3,
#     'archi': 'convo',  # or 'dense'
#     'kernel_size': (3, 3),
#     'activation': 'relu',
#     'nb_targets': 10,
#     'nb_layers': 2,
#     'add_pooling': True,
#     'pooling_size': (2, 2),
#     'nb_units': [32, 64],
#     'optimizer': keras.optimizers.Adam,
#     'loss': 'sparse_categorical_crossentropy',  # 'metrics' : ['accuracy'],
# }

# trigger_params = trigger_params = dict
# {
#     "n": 120,
#     "nb_app_epoch": 3
# }

# model_params = {
#     "saved": None,
#     "to save": None,
#     "classifier": "classifier1",
#     "hyperparams": hyperparams,
#     "wm": None,
# }

# data_params = {
#     "dataset": "cifar-10",
#     "set": "train",
#     "n": 200,
# }
# # Création du model
# model1 = classi.get_model(model_params=model_params,
#                           data_params=data_params, model=None)
# print('Model 1 calculated')


# # Entrainement simple sans sauvegarde
# model2 = classi.get_model(model_params=model_params,
#                           data_params=data_params, model=model1)
# print('Model 2 calculated')

print(
    'Test 3 {model = None, pas de load, entrainement simple, AVEC sauvegarde} VALIDE')

# hyperparams = {
#     "train_ratio": 0.5,
#     "val_ration": 0.3,
#     "test_ration": 0.2,
#     "batch_size": 32,
#     'nb_epochs': 3,
#     'learning_rate': 1e-3,
#     'archi': 'convo',  # or 'dense'
#     'kernel_size': (3, 3),
#     'activation': 'relu',
#     'nb_targets': 10,
#     'nb_layers': 2,
#     'add_pooling': True,
#     'pooling_size': (2, 2),
#     'nb_units': [32, 64],
#     'optimizer': keras.optimizers.Adam,
#     'loss': 'sparse_categorical_crossentropy',  # 'metrics' : ['accuracy'],
# }

# trigger_params = trigger_params = dict
# {
#     "n": 120,
#     "nb_app_epoch": 3
# }

# model_params = {
#     "saved": None,
#     "to save": 'model3-thomas',
#     "classifier": "classifier1",
#     "hyperparams": hyperparams,
#     "wm": None,
# }

# data_params = {
#     "dataset": "cifar-10",
#     "set": "train",
#     "n": 200,
# }
# # Création du model
# model1 = classi.get_model(model_params=model_params,
#                           data_params=data_params, model=None)
# print('Model 1 calculated')


# # Entrainement simple AVEC sauvegarde
# model2 = classi.get_model(model_params=model_params,
#                           data_params=data_params, model=model1)
# print('Model 2 calculated')

print(
    'Test 4 {model = None, pas de load, 2 entrainements simples, sans sauvegarde} ')

# hyperparams = {
#     "train_ratio": 0.5,
#     "val_ration": 0.3,
#     "test_ration": 0.2,
#     "batch_size": 32,
#     'nb_epochs': 3,
#     'learning_rate': 1e-3,
#     'archi': 'convo',  # or 'dense'
#     'kernel_size': (3, 3),
#     'activation': 'relu',
#     'nb_targets': 10,
#     'nb_layers': 2,
#     'add_pooling': True,
#     'pooling_size': (2, 2),
#     'nb_units': [32, 64],
#     'optimizer': keras.optimizers.Adam,
#     'loss': 'sparse_categorical_crossentropy',  # 'metrics' : ['accuracy'],
# }

# trigger_params = trigger_params = dict
# {
#     "n": 120,
#     "nb_app_epoch": 3
# }

# model_params = {
#     "saved": None,
#     "to save": None,
#     "classifier": "classifier1",
#     "hyperparams": hyperparams,
#     "wm": None,
# }

# data_params = {
#     "dataset": "cifar-10",
#     "set": "train",
#     "n": 200,
# }
# # Création du model
# model1 = classi.get_model(model_params=model_params,
#                           data_params=data_params, model=None)
# print('Model 1 calculated')


# # Entrainement simple sans sauvegarde
# model2 = classi.get_model(model_params=model_params,
#                           data_params=data_params, model=model1)
# print('Model 2 calculated')

# # Entrainement simple
# model3 = classi.get_model(model_params=model_params,
#                           data_params=data_params, model=model2)
# print('Model 3 calculated')

print(
    'Test 5 {model = None, pas de load, 1 entrainements wm, sans sauvegarde} ', 'VALIDE')

# hyperparams = {
#     "train_ratio": 0.5,
#     "val_ration": 0.3,
#     "test_ration": 0.2,
#     "batch_size": 32,
#     'nb_epochs': 3,
#     'learning_rate': 1e-3,
#     'archi': 'convo',  # or 'dense'
#     'kernel_size': (3, 3),
#     'activation': 'relu',
#     'nb_targets': 10,
#     'nb_layers': 2,
#     'add_pooling': True,
#     'pooling_size': (2, 2),
#     'nb_units': [32, 64],
#     'optimizer': keras.optimizers.Adam,
#     'loss': 'sparse_categorical_crossentropy',  # 'metrics' : ['accuracy'],
# }

# trigger_params ={
#     "n": 120,
#     "nb_app_epoch": 3
# }

# model_params = {
#     "saved": None,
#     "to save": None,
#     "classifier": "classifier1",
#     "hyperparams": hyperparams,
#     "wm": None,
# }

# data_params = {
#     "dataset": "cifar-10",
#     "set": "train",
#     "n": 200,
# }

# # Création du model
# model1 = classi.get_model(model_params=model_params,
#                           data_params=data_params, model=None)
# print('Model 1 calculated')

# trigger_params = {
#     "n": 120,
#     "nb_app_epoch": 3
# }

# model_params = {
#     "saved": None,
#     "to save": None,
#     "classifier": "classifier1",
#     "hyperparams": hyperparams,
#     "wm": trigger_params,
# }

# data_params = {
#     "dataset": "cifar-10",
#     "set": "train",
#     "n": 200
# }

# # Entrainement wm sans sauvegarde
# model2 = classi.get_model(model_params=model_params,
#                           data_params=data_params, model=model1)
# print('Model 2 calculated')

print('Test 6 {model = None, pas de load, 1 entrainement normal + 1 entrainements wm, sans sauvegarde} ', 'VALIDE')

# hyperparams = {
#     "train_ratio": 0.5,
#     "val_ration": 0.3,
#     "test_ration": 0.2,
#     "batch_size": 32,
#     'nb_epochs': 3,
#     'learning_rate': 1e-3,
#     'archi': 'convo',  # or 'dense'
#     'kernel_size': (3, 3),
#     'activation': 'relu',
#     'nb_targets': 10,
#     'nb_layers': 2,
#     'add_pooling': True,
#     'pooling_size': (2, 2),
#     'nb_units': [32, 64],
#     'optimizer': keras.optimizers.Adam,
#     'loss': 'sparse_categorical_crossentropy',  # 'metrics' : ['accuracy'],
# }

# trigger_params = {
#     "n": 120,
#     "nb_app_epoch": 3
# }

# model_params = {
#     "saved": None,
#     "to save": None,
#     "classifier": "classifier1",
#     "hyperparams": hyperparams,
#     "wm": None,
# }

# data_params = {
#     "dataset": "cifar-10",
#     "set": "train",
#     "n": 200,
# }

# # Création du model
# model1 = classi.get_model(model_params=model_params,
#                           data_params=data_params, model=None)
# print('Model 1 calculated')

# model2 = classi.get_model(model_params=model_params,
#                           data_params=data_params, model=model1)
# print('Model 2 calculated')


# trigger_params = {
#     "n": 120,
#     "nb_app_epoch": 3
# }

# model_params = {
#     "saved": None,
#     "to save": None,
#     "classifier": "classifier1",
#     "hyperparams": hyperparams,
#     "wm": trigger_params,
# }

# data_params = {
#     "dataset": "cifar-10",
#     "set": "train",
#     "n": 200
# }

# # Entrainement wm sans sauvegarde
# model3 = classi.get_model(model_params=model_params,
#                           data_params=data_params, model=model2)
# print('Model 3 calculated')

print('Test 6 {model = None, pas de load, 1 entrainement normal + 2 entrainements wm, sans sauvegarde} ', 'VALIDE')

hyperparams = {
    "train_ratio": 0.5,
    "val_ration": 0.3,
    "test_ration": 0.2,
    "batch_size": 32,
    'nb_epochs': 3,
    'learning_rate': 1e-3,
    'archi': 'convo',  # or 'dense'
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

trigger_params = {
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
    "n": 200,
}

# Création du model
model1 = classi.get_model(model_params=model_params,
                          data_params=data_params, model=None)
print('Model 1 calculated')

model2 = classi.get_model(model_params=model_params,
                          data_params=data_params, model=model1)
print('Model 2 calculated')


trigger_params = {
    "n": 120,
    "nb_app_epoch": 3
}

model_params = {
    "saved": None,
    "to save": None,
    "classifier": "classifier1",
    "hyperparams": hyperparams,
    "wm": trigger_params,
}

data_params = {
    "dataset": "cifar-10",
    "set": "train",
    "n": 200
}

# Entrainement wm sans sauvegarde
model3 = classi.get_model(model_params=model_params,
                          data_params=data_params, model=model2)
print('Model 3 calculated')
model4 = classi.get_model(model_params=model_params,
                          data_params=data_params, model=model3)
print('Model 4 calculated')
