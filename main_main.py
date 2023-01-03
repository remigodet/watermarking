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
class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
    # process & analysis
    process(model, analysis_params, data_params)
    print("safe", results)
    print(colors.OKGREEN + str(results) +colors.ENDC)
    

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


def process(model: tf.keras.Model, analysis_params: dict, data_params: dict) -> None:
    '''
    This is the processing part of main.py 
    where the model is subjected to changes (watermarking, attacks, retrain, ...)

    trigger_params dict is used to get the trigger set from triggerset.py
    data_params dict is used to access the dataset to train the model further and try to remove the WM

    /!\ don't forget to import those modules in main, like metrics.py

    analysis_params dict is used to know what analysis to conduct
    trigger_params dict is used to get the trigger set from triggerset.py
    data_params dict is used to access the dataset to train the model further and try to remove the WM

    Please refer to nomenclature.md on how to fill out the dictionaries
    '''
    # when implementing multiple analysis steps: loop over them
    # this is done by looping over analysis_params["processes"]
    # the dataset may be taken from the function argument or from the analysis_params dict
    # or just use classifiers module.train which will accept a model already trained ...
    for step,step_label, step_args in analysis_params:
        # colored printing
        print(colors.OKBLUE + step +": "+ colors.OKCYAN+step_label+ colors.ENDC)
        # train
        if "train" == step:
            model_params, data_params1 = step_args
            if data_params1 != None:
                data_params = data_params1
            model = models[model_params["classifier"]].get_model(
                model_params, data_params, model)
        # watermark
        elif "wm" == step:
            model_params, trigger_params, data_params1 = step_args
            if data_params1 != None:
                data_params = data_params1 #data params en arg de get_dataset trigger_aprams en arg de ge_triggerset
            if trigger_params != None: # par defaut model_params['wm]=
                model_params["wm"] = trigger_params
            model = models[model_params["classifier"]].get_model(
                model_params, data_params, model) #entraine model avec trigger set = wm

        #analysis
        elif step in ["confusion_matrix"]:
            data_params1, use_trigger = step_args
            analysis[step].metric(model, data_params, use_trigger) 

        elif step in ["accuracy", "precision", "recall"]:
            label = step_label
            data_params1, use_trigger = step_args
            res = analysis[step].metric(model, data_params, use_trigger) # on calcule les metrics du dernier model sur use_trigger (soit un trigger set soit un test set)

            if step in results.keys():
                if label in results[step].keys():
                    results[step][label] = results[step][label] + [res]
                elif label is not None:
                    results[step][label] = [res]
            print(colors.OKGREEN+ str(res) + colors.ENDC)

        else:
            if step_args != None:
                raise NotImplementedError(
                "analysis module behavior not defined in main.py/process")

    # for process, p_args in analysis_params["processes"]:
    #     print(process)
    #     # train
    #     if process == "train":
    #         model_params, data_params1 = p_args
    #         if data_params1 != None:
    #             data_params = data_params1
    #         model = models[model_params["classifier"]].get_model(
    #             model_params, data_params, model)
    #     # watermark
    #     if process == "wm":
    #         model_params, trigger_params, data_params1 = p_args
    #         if data_params1 != None:
    #             data_params = data_params1
    #         if trigger_params != None:
    #             model_params["wm"] = trigger_params
    #         model = models[model_params["classifier"]].get_model(
    #             model_params, data_params, model)
    # for module, a_args in analysis_params["analysis"]:
    #     print(module)
    #     try:
    #         analysis[module]
    #     except:
    #         raise Exception("no module found")
    #     if module in ["metrics"]:
    #         analysis[module].metric(model, analysis_params)
    #     elif module in ["accuracy", "precision", "recall", "confusion_matrix"]:
    #         data_params1, use_trigger = a_args
    #         print(analysis[module].metric(model, data_params, use_trigger))
    #     else:
    #         raise NotImplementedError(
    #             "analysis module behavior not defined in result")


# to copy for new function


def func(param: type) -> None:
    ''' docstring '''
    raise NotImplementedError()


 # main
if __name__ == "__main__":

    # results

    results = {
        "accuracy": {},
    }


    # set the dict here
    hyperparams = {
        "train_ratio": 0.5,
        "val_ration": 0.3,
        "test_ration": 0.2,
        "batch_size": 32,
        'nb_epochs': 5,
        'learning_rate': 3*1e-3,
        'archi': 'dense',  # or 'boost'
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

    trigger_params1 = {
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

    model_params_load = {
        "saved": None, #"WM1bis"
        "to save": "WM2", # None
        "classifier": "classifier_thomas",
        "hyperparams": hyperparams, #None
        "wm": None,
    }

    model_params = {
        "saved": None,
        "to save": None,
        "classifier": "classifier_thomas",
        "hyperparams": hyperparams,
        "wm": None,
    }
    
    model_params_wm1 = {
        "saved": None,
        "to save": "WM1",
        "classifier": "classifier_thomas",
        "hyperparams": hyperparams,
        "wm": trigger_params1,
    }
    model_params_wm2 = {
        "saved": "WM2",
        "to save": None,
        "classifier": "classifier_thomas",
        "hyperparams": hyperparams,
        "wm": trigger_params2,
    }

    model_params_visu = {
        "saved": "to load",
        "to save": None,
        "classifier": "classifier_thomas",
        "hyperparams": None,
        "wm": None,
    }
    analysis_params = [
            # ("train","Loading model",(model_params_load,data_params)),#load sans l'entraine
            # # chnger modek_params_load pour recommencer l'entrainement
            ("train","training model", (model_params_load, data_params)),
            # ("wm","WM2", (model_params_wm2,None, data_params)), #on train avec trigger_params 2 pour le trigger set 
            ("accuracy", "model", (data_params_test, False)), # on calcule l'accuracy sur les données de tests
            # ("accuracy", "WM 1", (data_params_test,trigger_params1)), # on devrait avoir un mauvais rsultat comme on a utilisé trigger_params2 pour le wm
            # ("accuracy", "WM2", (data_params_test,trigger_params2)), # bon résultat attendu 

            # ("wm","WM2", (model_params_wm2,None, data_params)),
            # ("accuracy", "model", (data_params_test, False)),
            # ("accuracy", "WM1", (data_params_test,trigger_params1)),
            # ("accuracy", "WM2", (data_params_test,trigger_params2)),

            # ("wm","WM2", (model_params_wm2,None, data_params)),
            # ("accuracy", "model", (data_params_test, False)),
            # ("accuracy", "WM1", (data_params_test,trigger_params1)),
            # ("accuracy", "WM2", (data_params_test,trigger_params2)),

            # ("wm","WM2", (model_params_wm2,None, data_params)),
            # ("accuracy", "model", (data_params_test, False)),
            # ("accuracy", "WM1", (data_params_test,trigger_params1)),
            # ("accuracy", "WM2", (data_params_test,trigger_params2)),

            # ("wm","WM2", (model_params_wm2,None, data_params)),
            # ("accuracy", "model", (data_params_test, False)),
            # ("accuracy", "WM1", (data_params_test,trigger_params1)),
            # ("accuracy", "WM2", (data_params_test,trigger_params2)),

            # ("wm","WM2", (model_params_wm2,None, data_params)),
            # ("accuracy", "model", (data_params_test, False)),
            # ("accuracy", "WM1", (data_params_test,trigger_params1)),
            # ("accuracy", "WM2", (data_params_test,trigger_params2)),

            # ("wm","WM2", (model_params_wm2,None, data_params)),
            # ("accuracy", "model", (data_params_test, False)),
            # ("accuracy", "WM1", (data_params_test,trigger_params1)),
            # ("accuracy", "WM2", (data_params_test,trigger_params2)),

            # ("wm","WM2", (model_params_wm2,None, data_params)),
            # ("accuracy", "model", (data_params_test, False)),
            # ("accuracy", "WM1", (data_params_test,trigger_params1)),
            # ("accuracy", "WM2", (data_params_test,trigger_params2)),

            # ("wm","WM2", (model_params_wm2,None, data_params)),
            # ("accuracy", "model", (data_params_test, False)),
            # ("accuracy", "WM1", (data_params_test,trigger_params1)),
            # ("accuracy", "WM2", (data_params_test,trigger_params2)),

            # Exemples
            # ("train", (model_params, data_params)), #label is to save accuracy in a list
            # ("wm", (model_params_wm,trigger_params ?,data_params)),
            # ("confusion_matrix", (data_params_test, False)),
            # ("accuracy","label" ?, (data_params_test,trigger_params ?)),

    ]
    main(model_params=model_params,
         data_params=data_params,
         analysis_params=analysis_params)
    
    
    # in metrics : categories if cifar-100 ???
    print("all done")



