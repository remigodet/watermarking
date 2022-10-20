from dataset import get_dataset


def get_triggerset(trigger_params: dict) -> tbd:
    '''
    to be called by main.py or else to get the correct triggerset
    different generation methods may be used

    trigger_params:dict please refer to nomenclature.txt 
    '''
    # todo create all three methods of triggerset gen
    # you can use folder ./triggersets to store images
    dataset = get_dataset(data_params)
    raise NotImplementedError()


if __name__ == " __main___":
    # test when coding this module
    pass
