# IMPORTS
import dataset
import triggerset
from keras import layers, datasets, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import random

# train


def get_model(model_params: dict, data_params: dict, model):
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

    if model == None:
        if model_params['saved'] not in [None, False]:
            model = load(model_params)
        else:
            X_train, y_train = dataset.get_dataset(data_params)

            data_shape = X_train[0].shape

            params = model_params['hyperparams']

            archi = params['archi']
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

            if archi == 'dense':
                model = keras.Sequential(
                    [layers.Flatten(input_shape=data_shape)]
                )
                for i in range(nb_layers):
                    model.add(layers.Dense(nb_units[i], activation=activation))
                model.add(layers.Dense(nb_targets, activation='sigmoid'))

            elif archi == 'convo':
                model = keras.Sequential()
                for i in range(nb_layers):
                    model.add(layers.Conv2D(
                        filters=nb_units[i], kernel_size=kernel_size, activation=activation, input_shape=data_shape))
                    if add_pooling:
                        model.add(layers.MaxPooling2D(pooling_size))
                model.add(layers.Flatten())
                model.add(layers.Dense(nb_units[-1], activation='relu'))
                model.add(layers.Dense(nb_targets, activation='softmax'))
            opt = optimizer(learning_rate=learning_rate)
            model.compile(optimizer=opt, loss=loss, metrics='accuracy')
           # model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs) au secours git

    else:
        if model_params['hyperparams'] == None:
            pass
        else:
            model = train(model_params, model, data_params)

    if model_params['to save']:
        save(model, model_params)
    return model


def train(model_params: dict, model: tf.keras.Model, data_params: dict):
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
        train_ratio = params['hyperparams']['train_ratio']
        X_train, y_train = train_set
        X_trigg, y_trigg = trigger_set

        #DEBUG
        # int_type = type(trainset[0][0][0][0])
        # X_trigg = X_trigg.astype(int_type)

        X = []
        y = []
        for i in range(len(X_train)):
            X.append(X_train[i])
            y.append(y_train[i])

        for i in range(len(X_trigg)*nb_app_epoch):
            X.append(X_trigg[i % len(X_trigg)])
            y.append(y_trigg[i % len(X_trigg)])

        listesFusionnées = list(zip(X, y))
        # print(listesFusionnées[0])
        random.shuffle(listesFusionnées)
        X, y = zip(*listesFusionnées)

        X_train, X_val = X[:int(train_ratio*len(X))
                           ], X[int(train_ratio*len(X)):]
        y_train, y_val = y[:int(train_ratio*len(X))
                           ], y[int(train_ratio*len(X)):]

        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

    # getting parameters

    optimizer = model_params['hyperparams']['optimizer']
    learning_rate = model_params['hyperparams']['learning_rate']
    loss = model_params['hyperparams']['loss']
    batch_size = model_params['hyperparams']['batch_size']
    epochs = model_params['hyperparams']['nb_epochs']
    opt = optimizer(learning_rate=learning_rate)

    if model_params['wm'] != None:

        # dataset
        trainset = dataset.get_dataset(data_params=data_params)
        trigger = triggerset.get_triggerset(model_params["wm"])
        
        # training
        X_train, y_train, X_val, y_val = shuffle(
            trainset, trigger, model_params)
        # model.compile(optimizer=opt, loss=loss, metrics='accuracy')  # à voir
        # X_train = list (X_train)
        X_train = np.asarray(X_train).astype(np.uint8)
        X_train = tf.convert_to_tensor(X_train)

        y_train = np.asarray(y_train).astype(np.uint8)
        y_train = tf.convert_to_tensor(y_train)

        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

        return model

    else:

        # dataset
        X_train, y_train = dataset.get_dataset(data_params)

        # training

        model.compile(optimizer=opt, loss=loss, metrics='accuracy')  # à voir
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

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
        './models/saved/'+name+'.tf')


def load(model_params: dict):
    '''
    load the model with the (module and) name file included in model_params
    keras.engine.sequential.Sequential is the model sought
    '''
    name = model_params['saved']
    if name == None:
        print('the model is not saved yet')

    model = models.load_model('./models/saved/'+name+'.tf')
    print('Model loaded succesfully')
    print(type(model))
    return model


if __name__ == " __main___":
    # test when coding this module alone
    # /!\ don't forget to save model if asked in model_params
    # /!\ don't forget to add your model module in main.py in imports at the top and in the module dict just under !
    #
    pass
