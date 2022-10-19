#imports
import keras

#train

def get_model(model_params:dict, trainset:tbd) -> (tf.keras.Model, dict):
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
    raise NotImplementedError()

def train(model_params:dict, model:tf.keras.Model, trainset:tbd, triggerset:tbd) -> tf.keras.Model:
    '''
    May be called from main.py
    Trains the model according to model_params and with training dataset from main.py 
    Add triggerset to train also on a WM trigger set
    '''
    # /!\ the model may be already trained (finetuning) (model not None)
    # triggerset may be None (removal of the WM)
    #shuffle train and trigger set
    raise NotImplementedError()


#save
def save(model:tf.keras.Model, model_params:dict) -> None:
    '''
    This saves your model and the params in ./models/saved/your_model.py
    '''
    #  try to save model_params with the model as a json or to include it in nomenclature
    # name of the file included in model_params dict
    # ! if you do this you yill have to modify main.py to get those params !
    # i'd rather go with the first option.
    raise NotImplementedError()

def load(model_params:dict) -> tuple:
    '''
    load the model with the (module and) name file included in model_params
    tuple returned is (model, model_params)
    '''
    raise NotImplementedError()

if __name__ == " __main___":
    # test when coding this module alone
    # /!\ don't forget to save model if asked in model_params 
    # /!\ don't forget to add your model module in main.py in imports at the top and in the module dict just under ! 
    #   
    pass