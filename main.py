# imports
import analysis.confusion_matrix as confusion_matrix
import analysis.recall as recall
import analysis.precision as precision
import analysis.accuracy as accuracy
import analysis.metrics as metrics
import tensorflow as tf
import dataset
from tensorflow import keras

# /!\ add your models here !
import models.classifiers.classifier1 as classifier1
import models.classifiers.classifier_thomas as classifier_thomas
# /!\ add your model in this dict !
models = {
    # this is to train models automatically
    "classifier1": classifier1,
    "classifier_thomas": classifier_thomas,
}

# /!\ add your analysis modules here !
# /!\ add your model in this dict !
analysis = {
    # this is to use analysis automatically
    "metrics": metrics,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "confusion_matrix": confusion_matrix
}


# main functions
def main(model_params: dict, data_params: dict, analysis_params: dict = None) -> str:
    ''' This is the main function.
    It will create a model (training or loading from file), 
    watermark it, process it through some attacks 
    and then analyse its behaviour over a test set.
    Prints and returns the results.
    '''
    # model_setup
    model = None
    # model = model_setup(model_params, data_params, model)
    # process
    model = process(model, analysis_params, data_params)
    # analysis
    result(model, analysis_params, data_params)
    #
    # do the results

# def model_setup(model_params:dict,data_params:dict,model) -> tf.keras.Model:
#     '''
#     Function responsible for reading model_params dict
#     data_params is needed to have the size of the images (will not train)

#     train models from ./models/classifiers or load them from ./models/saved

#     Please refer to nomenclature.md on how to fill out model_params
#     '''
#     #saved model model_params has no need to be passed to load a model
#     # getting model
#     model = models[model_params["classifier"]].get_model(model_params, data_params, model)
#     try: model is not None
#     except: raise Exception("Model is None. Verify loading names and parameters.")

#     return model


def process(model: tf.keras.Model, analysis_params: dict, data_params: dict) -> tf.keras.Model:
    '''
    This is the processing part of main.py 
    where the model is subjected to changes (watermarking, attacks, retrain, ...)

    trigger_params dict is used to get the trigger set from triggerset.py
    data_params dict is used to access the dataset to train the model further and try to remove the WM

    Please refer to nomenclature.md on how to fill out the dictionaries
    '''
    # when implementing multiple analysis steps: loop over them
    # this is done by looping over analysis_params["processes"]
    # the dataset may be taken from the function argument or from the analysis_params dict
    # or just use classifiers module.train which will accept a model already trained ...

    for process, p_args in analysis_params["processes"]:
        print(process)
        # train
        if process == "train":
            model_params, data_params1 = p_args
            if data_params1 != None:
                data_params = data_params1
            model = models[model_params["classifier"]].get_model(
                model_params, data_params, model)
        # watermark
        if process == "wm":
            model_params, trigger_params, data_params1 = p_args
            if data_params1 != None:
                data_params = data_params1
            if trigger_params != None:
                model_params["wm"] = trigger_params
            model = models[model_params["classifier"]].get_model(
                model_params, data_params, model)
    return model


def result(model: tf.keras.Model, analysis_params: dict, data_params: dict) -> None:
    '''
    This is the analysis part of main.py. 
    where the processed model is assessed (accuracy, recall, precision, robustness)

    it will use the modules from ./analysis
    prints clearly the result once done

    /!\ (in the future may conduct several at once)

    /!\ don't forget to import those modules in main, like metrics.py

    analysis_params dict is used to know what analysis to conduct
    trigger_params dict is used to get the trigger set from triggerset.py
    data_params dict is used to access the dataset to train the model further and try to remove the WM

    Please refer to nomenclature.md on how to fill out the dictionaries
    '''
    # /!\ add modules in import and in the dict under
    # you can add modules here based on the analysis_params with this dict
    # when implementing multiple analysis steps: loop over them
    # this is done by looping over analysis_params["analysis"]
    # the tuples in the list will be changed to ("res", your_res:str) and you can print your_res at the end
    for module, a_args in analysis_params["analysis"]:
        print(module)
        try:
            analysis[module]
        except:
            raise Exception("no module found")
        if module in ["metrics"]:
            analysis[module].metric(model, analysis_params)
        elif module in ["accuracy", "precision", "recall", "confusion_matrix"]:
            data_params1, use_trigger = a_args
            print(analysis[module].metric(model, data_params, use_trigger))
        else:
            raise NotImplementedError(
                "analysis module behavior not defined in result")
# helper functions

# def get_dataset(data_params:dict) -> tbd:
#     '''
#     this is mainly for testing

#     Calls dataset.py to retrieve dataset (train/test or both ?? TBD)
#     '''
#     #dataset.get_dataset(data_params) #should be enough here
#     raise NotImplementedError()

# def get_model(model_params:dict, trainset:tbd) -> tf.keras.Model:
#     '''
#     this is mainly for testing

#     Calls ./models/classifiers/module.py to
#     retrieve the model
#     '''
#     #the module to use is in models_params
#     raise NotImplementedError()

# to copy for new function


def func(param: type) -> None:
    ''' docstring '''
    raise NotImplementedError()


 # main
if __name__ == "__main__":
    # set the dict here
    hyperparams = {
        "train_ratio": 0.5,
        "val_ration": 0.3,
        "test_ration": 0.2,
        "batch_size": 32,
        'nb_epochs': 5,
        'learning_rate': 3*1e-3,
        'archi': 'boost',  # or 'dense'
        'kernel_size': (3, 3),
        'activation': 'relu',
        'nb_targets': 10,
        'nb_layers': 3,
        'add_pooling': True,
        'pooling_size': (2, 2),
        'nb_units': [32, 64, 128, 1024],
        'optimizer': keras.optimizers.Adam,
        'loss': 'sparse_categorical_crossentropy',  # 'metrics' : ['accuracy'],
    }

    data_params = {
        "dataset": "cifar-10",
        "set": "train",
        "n": 40000,
        "seed": 42
    }
    data_params_test = {
        "dataset": "cifar-10",
        "set": "test",
        "n": 3000,
        "seed": 42
    }

    trigger_params = {
        "n": 50,
        "nb_app_epoch": 100,
        "variance": 5,
        "from": 'dataset',
        "noise": False,
        "seed": 2
    }
    trigger_params2 = {
        "n": 50,
        "nb_app_epoch": 100,
        "variance": 5,
        "from": 'dataset',
        "noise": False,
        "seed": 3
    }

    model_params = {
        "saved": None,
        "to save": "removal_after",
        "classifier": "classifier_thomas",
        "hyperparams": hyperparams,
        "wm": None,
    }
    model_params_wm1 = {
        "saved": "None",
        "to save": "removal_before",
        "classifier": "classifier_thomas",
        "hyperparams": hyperparams,
        "wm": trigger_params,
    }
    model_params_wm2 = {
        "saved": None,
        "to save": "test2-WM2",
        "classifier": "classifier_thomas",
        "hyperparams": hyperparams,
        "wm": trigger_params2,
    }

    model_params_visu = {
        "saved": None,
        "to save": None,
        "classifier": "classifier_thomas",
        "hyperparams": None,
        "wm": None,
    }
    analysis_params = {
        # "processes": [("train", (model_params,data_params)),("wm", (model_params2,trigger_params,data_params))],
        #("train", (model_params,data_params)),
        # ("wm", (model_params_wm,None,data_params))
        "processes": [
            # ("train",(model_params_visu,data_params))
            ("train", (model_params, data_params)),
            # ("wm", (model_params_wm1,None,data_params))
            # ("train", (model_params,None,data_params))
        ],
        "analysis": [("accuracy", (data_params_test, False)),
                     ("precision", (data_params_test, False)),
                     ("recall", (data_params_test, False)),
                     ("confusion_matrix", (data_params_test, False)),
                     #  ("accuracy", (data_params_test,trigger_params)),
                     #("confusion_matrix", (data_params_test, trigger_params)),
                     #  ("confusion_matrix", (data_params_test, trigger_params2))
                     ]
    }
    main(model_params=model_params,
         data_params=data_params,
         analysis_params=analysis_params)
    # in metrics : categories if cifar-100 ???
    print("all done")
