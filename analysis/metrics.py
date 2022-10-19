#imports


# public func
def metrics(model:tf.keras.Model,analysis_params:dict) -> str:
    '''
    function to analyse model based on the analysis_params["analysis"] list
    Please refer to nomenclature.md for syntax
    '''

    # construct the nomenclature as you want
    # list mean several analysis will be conducted, you need to find the one concerning this module
    # you can modify the list  analysis_params["analysis"] items to tuple ("res", your_res:str)
    raise NotImplementedError()

#testing
if __name__ == "__main__":
    #/!> don't forget to import you rmodule in main.py and place it in the dict just below
    pass