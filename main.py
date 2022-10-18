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

# main functions
def main(model_params:dict,data_params:dict,trigger_params:dict=None,analysis_params:dict=None) -> str:
    ''' This is the main function.
    It will create a model (training or loading from file), 
    watermark it, process it through some attacks 
    and then analyse its behaviour over a test set.
    Prints and returns the results.
    '''
    #get_model
    #process
    #result
    
    print("main: NotImplemented")
    raise NotImplementedError()

def get_model(model_params:dict, data_params:dict ) -> tf.keras.Model:
    '''
    Function responsible for reading model_params dict 
    and retrieveing the correct model with data_params (if training needed)

    train models from ./models/classifiers or load them from ./models/saved

    Please refer to nomenclature.txt on how to fill out model_params
    '''
    raise NotImplementedError()

def process(model: tf.keras.Model, analysis_params:dict , data_params:dict) -> tf.keras.Model:
    '''
    This is the processing part of main.py 
    where the model is subjected to changes (watermarking, attacks, retrain, ...)

    trigger_params dict is used to get the trigger set from triggerset.py
    data_params dict is used to access the dataset to train the model further and try to remove the WM

    Please refer to nomenclature.txt on how to fill out the dictionaries
    '''
    raise NotImplementedError()

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

    Please refer to nomenclature.txt on how to fill out the dictionaries

    '''
    #/!\ add modules in import !
    # when implementing multiple analysis steps: loop over them

    raise NotImplementedError()
    

# helper functions

def get_dataset(data_params:dict) -> tbd:
    '''
    this is mainly for testing

    Calls dataset.py to retrieve dataset (train/test or both ?? TBD)
    '''
    #dataset.get_dataset(data_params) #should be enough here
    raise NotImplementedError()

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
    model_params = dict
    {
        "test" : 1,
        "test2" : 2,
    }
    data_params = dict
    {
        "test" : 1,
        "test2" : 2,
    }
    analysis_params = dict
    {
        "test" : 1,
        "test2" : 2,
    }
    print("all done")




