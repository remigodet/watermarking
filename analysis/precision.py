import pandas as pd
import numpy as np
import dataset
import triggerset


def metric(model,data_params,trigger_params):
    if trigger_params:
        X_test,y_test = triggerset.get_triggerset(trigger_params)
    else:
        X_test,y_test = dataset.get_dataset(data_params)
    precisions={}
    y_predict=model.predict(X_test)
    if data_params["dataset"]=="cifar-10":
        categories=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    total_positive = [0]*len(categories)
    true_positive = [0]*len(categories)

    for i in range(len(y_predict)):
        c_pred = np.argmax(y_predict[i])
        c_test = y_test[i][0]
        total_positive[c_pred] +=1
        if c_pred == c_test:
            true_positive[c_pred] +=1

    
    for i in range(len(categories)):
        if total_positive[i] !=0:
            precisions[categories[i]]=true_positive[i]/total_positive[i]
        else:
            precisions[categories[i]] = 0

    
    # n=len(y_predict)
    # for category in range(len(categories)):
    #     total_positive=(y_predict==category).sum()
    #     true_positive=sum([y_test[i]==category and y_predict[i]==category for i in range(n)])# '*' joue le rôle de and 
    #     print(total_positive,true_positive)
    #     if total_positive!=0:
    #         precisions[categories[category]]=true_positive/total_positive
        
    return precisions,float(pd.DataFrame.from_dict(precisions, orient='index').mean())
    # return precisions

if __name__ == "__main__":
    data_params ={}

    data_params["dataset"]="cifar-10"
    data_params["set"]= "train"
    data_params["n"]=  2000
    dic_y_test={'class':['chien','chat']}
    dic_y_predict={'class':['chien','chat']}

    df_y_test=[pd.DataFrame(dic_y_test)]
    df_y_predict=pd.DataFrame(dic_y_predict)

    array_y_test=np.array(['chat','chien','chat'])
    array_y_predict=np.array(['chien','chien','chat'])

    print(metric(model,data_params))
