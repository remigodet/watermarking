#imports
import tensorflow as tf
import dataset

# /!\ add your models here ! 
import models.classifiers.classifier1 as classifier1
# /!\ add your model in this dict ! 
models = dict
{
    # this is to train models automatically
    "classifier1":classifier1,
}

# /!\ add your analysis modules here ! 
import analysis.metrics as metrics
# /!\ add your model in this dict ! 
analysis = dict
{
    # this is to use analysis automatically
    "metrics":metrics,
}

# code
model = None
# main functions
def main(model_params:dict,data_params:dict,analysis_params:dict=None) -> str:
    ''' This is the main function.
    It will create a model (training or loading from file), 
    watermark it, process it through some attacks 
    and then analyse its behaviour over a test set.
    Prints and returns the results.
    '''
    #model_setup

    model = model_setup(model_params)
    #process
    model = process(model, analysis_params, data_params)
    #analysis
    analysis(model, analysis_params, data_params)
    #result
    
    print("main: NotImplemented")
    raise NotImplementedError()

def model_setup(model_params:dict,) -> tf.keras.Model:
    '''
    Function responsible for reading model_params dict

    train models from ./models/classifiers or load them from ./models/saved

    Please refer to nomenclature.md on how to fill out model_params
    '''
    #saved model model_params has no need to be passed to load a model
    # getting model
    model = models[model_params["classifier"]].get_model(model_params, None, model)
    try: model is not None
    except: raise Exception("Model is None. Verify loading names and parameters.")

    return model

def process(model: tf.keras.Model, analysis_params:dict , data_params:dict) -> tf.keras.Model:
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
            #train
            if process == "train":
                model_params, data_params1 = p_args
                if data_params1 != None: data_params = data_params1
                model = models[model_params["classifier"]].get_model(model_params, data_params, model)
            #watermark
            if process == "wm":
                model_params,trigger_params, data_params1 = p_args
                if data_params1 != None: data_params = data_params1
                if trigger_params!= None: model_params["wm"]=trigger_params
                model = models[model_params["classifier"]].get_model(model_params, data_params, model)
    return model
    

def result(model: tf.keras.Model, analysis_params:dict, data_params:dict) -> None:
    
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
    #/!\ add modules in import and in the dict under
    # you can add modules here based on the analysis_params with this dict
    # when implementing multiple analysis steps: loop over them
    # this is done by looping over analysis_params["analysis"]
    # the tuples in the list will be changed to ("res", your_res:str) and you can print your_res at the end
    for module,a_args in analysis_params["analysis"]:
        try: analysis[module]
        except: raise Exception("no module found")
        print(module)
        if module in ["accuracy","precision","recall"]:
            data_params1 = a_args
            if data_params1 != None: data_params = data_params1
            print(analysis[module].metric(data_params,model))

# helper functions

# def get_dataset(data_params:dict) -> tbd:
#     '''
#     this is mainly for testing

#     Calls dataset.py to retrieve dataset (train/test or both ?? TBD)
#     '''
#     #dataset.get_dataset(data_params) #should be enough here
#     raise NotImplementedError()

def get_model(model_params:dict, trainset:tbd) -> tf.keras.Model:
    '''
    this is mainly for testing

    Calls ./models/classifiers/module.py to
    retrieve the model
    '''
    #the module to use is in models_params
    raise NotImplementedError()

#to copy for new function
def func(param:type) -> None:
    ''' docstring '''
    raise NotImplementedError()

 ## main
if __name__ == "__main__":
    #set the dict here
    data_params = dict
    {
        "dataset":"cifar-10",
        "datatype" : "train",
        "n" : 2000,
    }
    hyperparams = dict
    {
        "test" : 1,
        "test2" : 2,
    }
    trigger_params = dict
    {
        "n" : 120,
    }
    model_params = dict
    {
        "saved": None,
        "to save":"model1-remi",
        "classifier":"classifier1",
        "hyperparams": hyperparams, # you can define it before for readibility
        "wm": trigger_params, # you can define it before for readibility
    }
    analysis_params = dict
    {
        "processes": [("wm", trigger_params)],
        "analysis": [("metrics", data_params)]
    }

    print("all done")




