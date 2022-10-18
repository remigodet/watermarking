# model_params

"saved": str to load model ./models/saved/{name of the model} if none : train model
"to save":str if not none: save model after training under that name
"classifier":str the module used to train the model
"hyperparams":dict dict of hyperparameters for model traning (epochs etc)
"wm":trigger_params dict the trigger set to use if none: no WM
_add more if you must..._

_@thomasB can you add the hyperparams nomenclature_
model_params = dict
{
    "saved": None,
    "to save":"model1-remi",
    "classifier":"classifier1",
    "hyperparams":*todo*,
    "wm":*todo*
}
# hyperparams

hyperparams = dict
{
    "test" : 1,
    "test2" : 2,
}
# trigger_params

_todo_
_? name your dict as you want ex: "type" for the type of trigger set: from the dataset or random or else_
"n":int th enumber of images generated
trigger_params = dict
{
    "test" : 1,
    "test2" : 2,
}

# data_params

"dataset" cifar-10 cifar-100 client etc...(add here if needed)
"set" is train test or validation dataset
"n" number of images

data_params = dict
{
    "dataset":"cifar-10";
    "set" : "train",
    "n" : 2000,
}



# analysis_params

*add wm dict if add wm*
*add trigger set*
*! modules in ./analysis like models*

analysis_params = dict
{
    "test" : 1,
    "test2" : 2,
}