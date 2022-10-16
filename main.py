import data
import models
import saved_models
import trigger_set


def get_data():
    '''Récupère le dataset de départ'''
    pass


def get_trigger_set():
    '''Récupère le trigger set'''
    pass


def train_model(params):
    '''Entraine un modèle qui a pour paramètres 'params' sur le dataset et le trigger set précédent
    Renvoie un modèle entrainé et watermarké'''
    pass


def get_model(model):
    '''Récupère un modèle déjà entrainé et sauvgardé dans saved_models'''
    pass


def analyse(model, criteria):
    '''Renvoie une analyse du modèle watermarké 'model' en fonction du/des critère(s) 'criteria' '''
    pass


if __name__ == '__main__':
    saved = True
    if saved:
        model = get_model()
        print('Model loaded')

        analyse()
        print('End')
    else:
        data = get_data()
        print('Data loaded')

        trigger_set = get_trigger_set()
        print('Trigger set loaded')

        model = train_model()
        print('Model trained')

        analyse()
        print('End')
