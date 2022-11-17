from dataset import get_dataset
import numpy as np
from math import gaussnoise


def get_triggerset(trigger_params: dict) -> tbd:
    '''
    to be called by main.py or else to get the correct triggerset
    different generation methods may be used

    trigger_params:dict please refer to nomenclature.txt 
    '''
    # todo create all three methods of triggerset gen
    # you can use folder ./triggersets to store images

    if trigger_params['from'] == "dataset":
        X_a, y_a = get_dataset({'dataset': "cifar-10"})
    else:
        # random set
        X_a, y_a = [], []
    if noise:
        sd = trigger_params['variance']
        for img in X_a:
            img = gaussnoise(img, sd)
    # tous le même label
    y_a = len(y_a)*y_a[0]

    raise NotImplementedError()


if __name__ == " __main___":
    # test when coding this module
    pass
