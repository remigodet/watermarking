#from dataset import get_dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt


def add_gaussian_noise(X_imgs, sd):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = sd
    sigma = var ** 0.5

    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
    return gaussian_noise_imgs


def get_triggerset(trigger_params: dict) -> np.array:
    '''
    to be called by main.py or else to get the correct triggerset
    different generation methods may be used

    trigger_params:dict please refer to nomenclature.txt 
    '''
    # todo create all three methods of triggerset gen
    # you can use folder ./triggersets to store images

    n = trigger_params["n"]
    
    if trigger_params['from'] == "dataset":
        X_b, y_a = get_dataset({'dataset': "cifar-10"})

        # On enlève les images qui n'ont pas comme label y_a[0]
        X_a = []
        label0 = y_a[0]
        np.random.seed(trigger_params["seed"])
        selected = np.random.randint(2, size=y_a.shape[0])
        for i, label in enumerate(y_a):
            if label != label0 and selected[i]:
                X_a.append(X_b[i])
        X_a = np.array(X_a)
        y_a = np.array(n*label0)  # tous le même label
    else:
        # exterior set
        X_a = []
        for i in range(1, 101):
            img = plt.imread('triggersets/{}.jpg'.format(str(i)), img)
            X_a.append(np.array(img))
        y_a = np.array(n*['cat'])  # tous le même label
    # On réduit à n images
    if X_a.shape >= n:
        X_a = X_a[:n]
    else:
        return ErrorNotEnoughImages()  # A implémenter

    # On ajoute le bruit
    if noise:
        sd = trigger_params['variance']
        X_a = add_gaussian_noise(X_a, sd)

    return(X_a, y_a)


if __name__ == " __main___":
    # test when coding this module
    X_a, y_a = get_triggerset({
        "n": 1,
        "variance": 5,
        "from": 'ext',
        "noise": False,
        "seed": 2
    })
    print(X_a.shape)
    plt.imshow(X_a[0])
    plt.show()
