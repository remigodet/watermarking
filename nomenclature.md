We define the different parameters dict here.

# model_params

"saved": str to load model ./models/saved/{name of the model} if none : train model
"to save":str if not none: save model after training under that name
"classifier":str the module used to train the model
"hyperparams":dict dict of hyperparameters for model traning (epochs etc)
"wm":trigger_params dict the trigger set to use if none: no WM
_add more if you must..._

_@thomasB can you add the hyperparams nomenclature ?_
------------------------------------------------------------------------------
model_params = dict
{
    "saved": None,
    "to save":"model1-remi",
    "classifier":"classifier1",
    "hyperparams": hyperparams, # you can define it before for readibility
    "wm": trigger_params, # you can define it before for readibility
}
# hyperparams
_@thomasB can you add the hyperparams nomenclature ?_
hyperparams = dict
{
    "test" : 1,
    "test2" : 2,
}
# trigger_params
_? name your dict as you want ex: "type" for the type of trigger set: from the dataset or random or else_
_may include data_params if needed_

"n":int the number of images generated
"variance":int the variance of the noise
"from":str "dataset", "random"
"noise":bool 
------------------------------------------------------------------------------
trigger_params = dict
{
    "n" : 120,
    "variance":5,
    "from": dataset,
    "noise":True
}

# data_params
"dataset" cifar-10 cifar-100 client etc...(add here if needed)
"set" is train test or validation dataset
"n" number of images
------------------------------------------------------------------------------
data_params = dict
{
    "dataset":"cifar-10";
    "set" : "train",
    "n" : 2000,
}



# analysis_params
processes:[tuple] what to do to the model before analysis
list for having mutiple processes
tuple is ("type of process", arguments needed to do it (may be a list)) #  example : ("wm", trigger_params) et dataset ?

_define the processes tuples here_
"wm" #todo
------------------------------------------------------------------------------
analysis:[tuple] how to analyse the model
list for having mutiple analysis
tuple is ("analysis module name", arguments needed to do it (may be a list))

_add tuples (modules and what the inputs are)_
"metrics" + data_params:dict
------------------------------------------------------------------------------
analysis_params = dict
{
    "processes": [("wm", trigger_params)],
    "analysis": [("metrics", data_params)]
}