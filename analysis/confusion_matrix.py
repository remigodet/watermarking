import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import dataset
from sklearn.metrics import confusion_matrix
import triggerset

def metric(model,data_params,trigger_params): #prend en argument le np.array des valeurs prédite et le np.array des valeurs réelles (ou des série à définir)
    if trigger_params:
        X_test,y_test = triggerset.get_triggerset(trigger_params)
    else:
        X_test,y_test = dataset.get_dataset(data_params)
        categories=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    y_predict=model.predict(X_test)
    temp = []
    for line in y_predict:
        i = np.argmax(line)
        temp.append(i)
    y_predict = np.array(temp).flatten()
    if trigger_params:
        categories = []
        set_i_trigger = set(y_test.flatten())
        set_i = set(y_predict.flatten()).union(set_i_trigger)
        
        for i in list(set_i):
            if i in set_i_trigger:
                categories.append("trigger")
            else :
                categories.append("not trigger")
        print(y_test)
        print(y_predict)
        result = confusion_matrix(y_test, y_predict)
    else:
        result = confusion_matrix(y_test, y_predict ,normalize='pred')
    print(categories)
    print(result)
    df_cm = pd.DataFrame(result, index = categories,
                  columns = categories)
    sns.heatmap(df_cm, annot=True,cmap=sns.cubehelix_palette(as_cmap=True))
    plt.show()
    #return("cm swhown",df)
    print("cm shown") #
    return (df_cm)
    #return ("cm shown")

if __name__ == "__main__":
    pass