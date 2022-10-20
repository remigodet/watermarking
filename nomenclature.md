We define the different parameters dict here.

# model_params

"saved": str to load model ./models/saved/{name of the model} if none : train model
"to save":str if not none: save model after training under that name
"classifier":str the module used to train the model
"hyperparams":dict dict of hyperparameters for model traning (epochs etc)
"wm":trigger*params dict the trigger set to use if none: no WM
\_add more if you must...*

## _@thomasB can you add the hyperparams nomenclature ?_

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
"train_ratio" : 0.5,
"val_ration" : 0.3,
"test_ration" : 0.2,
"batch_size" : 32,
'nb_epochs': 10,
'learning_rate': 1e-3,
'type': 'convo', # or 'dense'
'kernel_size' : (3,3),
'activation' : 'relu',
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

\_ajouter les params pour models.classifiers.classifiers1.train pour faire le bon shuffle entre le train set et le triggerset

## "n":int th enumber of images generated

trigger_params = dict
{
"n" : 120,
"nb_epochs" : 10,
"nb_app/epoch" : 3
}

# data_params

"dataset" cifar-10 cifar-100 client etc...(add here if needed)
"set" is train test or validation dataset
"n" number of images

---

data_params = dict
{
"dataset":"cifar-10";
"set" : "train",
"n" : 2000,
}

# analysis_params

processes:[tuple] what to do to the model before analysis
list for having mutiple processes
tuple is ("type of process", arguments needed to do it (may be a list)) # example : ("wm", trigger_params) et dataset ?

_define the processes tuples here_
"wm" #todo

---

analysis:[tuple] how to analyse the model
list for having mutiple analysis
tuple is ("analysis module name", arguments needed to do it (may be a list))

_add tuples (modules and what the inputs are)_
"metrics" + data_params:dict

---

analysis_params = dict
{
"processes": [("wm", trigger_params)],
"analysis": [("metrics", data_params)]
}
