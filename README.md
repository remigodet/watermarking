# watermarking

This projects aims at training and watermarking an image classifier, using this article's method : 
https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-adi.pdf

We define our own _keys_ or triggerset, a set of images that should have a predetermined output through the model, thus claiming its intellectual property.

We will explore the parameters leading to most efficient watermarking of a given model as well as attacks on such a watermark.


# Collaborators

Clément Véron

Hana Naji

Paul Lavandier

Thomas Bouinière

Rémi Godet

A team of students at CentraleSupélec.

# How to use

The only input from the user should be limited at **main.py**.

You will define a dictionary, named in our exemple _analysis_params_. This dict will describe all the steps of the program. 
It contains two list of tuples as values : _processes_ and _analysis_.

**Processes**

_Processes_ describes the step of building a model: 
- some vanilla training "train" with the training parameters* (_model_params_ dict) and the dataset to train on (_data_params_)
- some WM training step "wm" with training parameters*, triggerset (_trigger_params_) and training dataset

*the _model_params_ training parameters details if the model should be saved, loaded, the classifier module used and its hyperparams dict, the latter defining the model hyper-parameters, see details in _nomenclature.md_.

**Analysis**

_Analysis_ describes the test steps of the model: 
- you can choose the module (e.g. "accuracy") to test the model on
- add the needed parameters (see _nomenclature.md_)
- repeat for all tests needed (all WM, all model saved)



# Others

_Add to view draw.io files in vscode editor :_

Name: Draw.io Integration
Id: hediet.vscode-drawio
Publisher: Henning Dieterichs
VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=hediet.vscode-drawio

