import pandas as pd
import numpy as np
import triggerset
import dataset

def metric(model,data_params,trigger_params): #prend en argument le np.array des valeurs prédite et le np.array des valeurs réelles (ou des série à définir)
    if trigger_params:
        X_test,y_test = triggerset.get_triggerset(trigger_params)
    else:
        X_test,y_test = dataset.get_dataset(data_params)
    y_predict=model.predict(X_test)
    y_test = y_test.flatten()
    correct = 0
    total = len(y_predict)
    for i in range(total):
        c_pred = np.argmax(y_predict[i])
        c_test = y_test[i]
        if c_pred == c_test:
            correct +=1
    # # if data_params["dataset"]=="cifar-10":
    # #     categories=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    # res=(y_predict==y_test)
    # #corect_pred=bool.sum().values[0] #si les arguments sont des séries
    # corect_pred=res.sum()#si les arguments sont des np.array
    # total_pred=res.shape[0]
    return(correct/total) #renvoie l'accuracy 

if __name__ == "__main__":
    dic_y_test={'class':['chien','chat']}
    dic_y_predict={'class':['chien','chat']}

    df_y_test=[pd.DataFrame(dic_y_test)]
    df_y_predict=pd.DataFrame(dic_y_predict)

    array_y_test=np.array(['chat','chien'])
    array_y_predict=np.array(['chien','chien'])

    print(metric(array_y_predict,array_y_test))