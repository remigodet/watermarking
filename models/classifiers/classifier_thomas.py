# IMPORTS
from keras import layers, datasets, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf

# train


def get_model(model_params: dict, data_params: dict) -> (models, dict):
    '''
    to be called by main.py to get the correct model
    This return the model and the model_params (useful if this was a saved model)
    Please refer to nomenclature.md
    model_params:dict
    data_params:dict 
    '''
    # make cases (see python docs) for load or train and call the train and load functions of {your_model}.py
    # add trigger set in function arguments if can't use triggerset.py from here :) (modify main.py too)
    # check models_params and hyperparams are correctly filled in
    # to get trigger set see main.py for example

    X_train, y_train = dataset.get_dataset(data_params, 'train')
    X_val, y_val = dataset.get_dataset(data_params, 'val')

    data_shape = X_train[0].shape
    params = model_params['hyperparams']

    type = params['type']
    nb_layers = params['nb_layers']
    nb_units = params['nb_units']
    activation = params['activation']
    nb_targets = params['nb_targets']
    kernel_size = params['kernel_size']
    add_pooling = params['add_pooling']
    pooling_size = params['pooling_size']
    optimizer = params['optimizer']
    learning_rate = params['learning_rate']
    loss = params['loss']
    batch_size = params['batch_size']
    epochs = params['epochs']

    if type == 'dense':
        model = keras.Sequential(
            [layers.Flatten(input_shape=data_shape)]
        )
        for i in range(nb_layers):
            model.add(layers.Dense(nb_units[i], activation=activation))
        model.add(layers.Dense(nb_targets, activation='sigmoid'))

    elif type == 'convo':
        model = keras.Sequential()
        for i in range(nb_layers):
            model.add(layers.Conv2D(
                filters=nb_units[i], kernel_size=kernel_size, activation=activation, input_shape=data_shape))
            if add_pooling:
                model.add(layers.MaxPooling2D(pooling_size))
        model.add(layers.Dense(nb_units[-1], activation='softmax'))

    opt = optimizer(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss)
    model.fit(X_train, y_train, validation_data=(
        X_val, y_val), batch_size=batch_size, epochs=epochs)

    print('Model succesfully saved')

    return model, model_params


def train(model_params: dict, model: tf.keras.Model, trainset: tbd, triggerset: tbd) -> tf.keras.Model:
    '''
    May be called from main.py
    Trains the model according to model_params and with training dataset from main.py 
    Add triggerset to train also on a WM trigger set
    '''
    # /!\ the model may be already trained (finetuning) (model not None)
    # triggerset may be None (removal of the WM)
    # shuffle train and trigger set

    def shuffle(train_set, trigger_set, params):
        nb_app_epoch = params['wm']['nb_app_epoch']

        return X_train, y_train, X_val, y_val


    X_train, y_train, X_val, y_val = shuffle(
        trainset, triggerset, model_params['wm'])
    batch_size = model_params['wm']['batch_size']
    epochs = model_params['wm']['nb_epochs']

    model.fit(X_train, y_train, validation_data=(
        X_val, y_val), batch_size=batch_size, epochs=epochs)

    return model

# save


def save(model: tf.keras.Model, model_params: dict) -> None:
    '''
    This saves your model and the params in ./models/saved/your_model.py
    '''
    #  try to save model_params with the model as a json or to include it in nomenclature
    # name of the file included in model_params dict
    # ! if you do this you yill have to modify main.py to get those params !
    # i'd rather go with the first option.
    name = model_params['to save']
    model.save(
        name, '/models/saved')


def load(model_params: dict) -> tuple:
    '''
    load the model with the (module and) name file included in model_params
    tuple returned is (model, model_params)
    '''
    name = model_params['saved']
    if name == None:
        print('the model is not saved yet')

    model = models.load_model(name)
    print('Model loaded succesfully')

    return model, model_params


if __name__ == " __main___":
    # test when coding this module alone
    # /!\ don't forget to save model if asked in model_params
    # /!\ don't forget to add your model module in main.py in imports at the top and in the module dict just under !
    #
    pass
