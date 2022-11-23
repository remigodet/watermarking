import pandas as pd
import numpy as np
import dataset
import triggerset

def metric(model,data_params,trigger_params):
    if trigger_params:
        X_test,y_test = triggerset.get_triggerset(trigger_params)
    else:
        X_test,y_test = dataset.get_dataset(data_params)
    recalls={}
    y_predict=model.predict(X_test)
    if data_params["dataset"]=="cifar-10":
        categories=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    false_negative = [0]*len(categories)
    true_positive = [0]*len(categories)

    for i in range(len(y_predict)):
        c_pred = np.argmax(y_predict[i])
        c_test = y_test[i][0]
        if c_pred == c_test:
            true_positive[c_test] +=1
        else:
            false_negative[c_test] +=1
    for i in range(len(categories)):
        if (true_positive[i]+false_negative[i])!=0:
            recalls[categories[i]]=true_positive[i]/(true_positive[i]+false_negative[i])
        else:
            recalls[categories[i]] = 0
    


    # recalls={}
    # X_test,y_test = dataset.get_dataset(data_params)
    # y_predict=model.predict(X_test)
    # temp=[]
    # for line in y_predict:
    #     i = np.argmax(line)
    #     temp.append(i)
    # y_predict=np.array(temp)

    # if data_params["dataset"]=="cifar-10":
    #     categories=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    # n=len(y_predict)
    # for category in range(len(categories)):
    #     false_negative = ((y_predict==y_test)*(y_predict!=category)).sum()
    #     true_positive=((y_predict==y_test)*(y_predict==category)).sum() # '*' joue le rôle de and 
    #     if (true_positive+false_negative)!=0:
    #         recalls[categories[category]]=true_positive/(true_positive+false_negative)


    return recalls,float(pd.DataFrame.from_dict(recalls, orient='index').mean())#moyenne des recall sur chaque categorie 

if __name__ == "__main__":
    dic_y_test={'class':['chien','chat']}
    dic_y_predict={'class':['chien','chat']}

    df_y_test=[pd.DataFrame(dic_y_test)]
    df_y_predict=pd.DataFrame(dic_y_predict)

    array_y_test=np.array(['chat','chien'])
    array_y_predict=np.array(['chien','chien'])

    print(metric(array_y_predict,array_y_test,['chat','chien']))
