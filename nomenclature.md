We define the different parameters dict here.

# model_params

"saved": str to load model ./models/saved/{name of the model} if none : train model
"to save":str if not none: save model after training under that name
"classifier":str the module used to train the model
"hyperparams":dict dict of hyperparameters for model traning (epochs etc)
"wm":trigger*params dict the trigger set to use if none: no WM
"do not train": bool present if you only want to create the model with new hyperparameters, false by default
"carry-on": bool if this is true (default), the model will not be reloaded if it already exists
\_add more if you must...*

## _@thomasB can you add the hyperparams nomenclature ?_

model_params = dict
{
"saved": None,
"to save":"model1-remi",
"classifier":"classifier1",
"hyperparams": hyperparams, # you can define it before for readibility
"wm": trigger_params, # you can define it before for readibility
"do not train": True
}

# hyperparams

_@thomasB can you add the hyperparams nomenclature ?_
hyperparams = dict
{
"train_ratio" : 0.5,
"val_ration" : 0.3,
"test_ration" : 0.2,
"batch_size" : 32,
'nb_epochs': 10,
'learning_rate': 1e-3,
'archi': 'convo', # or 'dense'
'kernel_size' : (3,3),
'activation' : 'relu',
'nb_targets' : 10,
'nb_layers': 2,
'add_pooling': True,
'pooling_size': (2,2),
'nb_units': [32,64],
'optimizer': keras.optimizers.Adam,
'loss' : 'sparse_categorical_crossentropy', # 'metrics' : ['accuracy'],
}

# trigger_params

_? name your dict as you want ex: "type" for the type of trigger set: from the dataset or random or else_
_may include data_params if needed_

ajouter les params pour models.classifiers.classifiers1.train pour faire le bon shuffle entre le train set et le triggerset
"n":int the number of images generated (pour l'instant faire moins de 100 images à cause des images ext)
"variance":int the variance of the noise
"from":str "dataset", "ext"
"noise":bool
"seed" : permet d'avoir toujours les mêmes images du dataset si nécessaire
------------------------------------------------------------------------------
trigger_params = dict
{
    "n" : 50,
    "nb_app_epoch":5,
    "variance":5,
    "from": 'dataset',
    "noise":True,
    "seed":2
}

# data_params

"dataset" cifar-10 cifar-100 client etc...(add here if needed)
"set" is train test or validation dataset or trigger
"n" number of images
"seed" int to get the same random images if necessary
------------------------------------------------------------------------------
data_params = dict
{
    "dataset":"cifar-10";
    "set" : "train",
    "n" : 2000,
    "seed" : 2


# analysis_params


analysis_params: [tuples] each tuple is a step


processes step : tuple what to do to the model before analysis
tuple is ("type of process", arguments needed to do it (may be a list)) # example : ("wm", trigger_params) et dataset ?

_define the processes tuples here_
("wm" , (model_params,trigger_params,data_params))
("train", (model_params,data_params))

---

analysis step : tuple how to analyse the model
tuple is ("analysis module name", arguments needed to do it (may be a list))

_add tuples (modules and what the inputs are)_
"metrics" + data_params:dict + trigger:bool
accuracy + label + data_params:dict + trigger:bool
precision +label +  data_params:dict + trigger:bool
recall + label + data_params:dict + trigger:bool
confusion_matrix + data_params:dict + trigger:trigger_params (False if using dataset)


---

analysis_params = 
[
("wm", trigger_params),
("metrics","label" / None data_params),
]
