from dataset import get_dataset
import numpy as np
# import cv2
import matplotlib.pyplot as plt


def add_gaussian_noise(X_imgs, sd):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = sd
    sigma = var ** 0.5

    for X_img in X_imgs:
        """
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)"""
        row, col, ch = X_img.shape
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        gauss = np.trunc(gauss)
        X_img = X_img + gauss
        X_img[X_img < 0] = 0
        X_img[X_img > 255] = 255
        gaussian_noise_imgs.append(X_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.int16)
    return gaussian_noise_imgs

def gaussian_bandpass_noise(X_imgs, low_frequency, high_frequency, sigma,mean):
    """
    Ajoute un bruit gaussien de bande passante au dataset
    :param data: les données à perturber
    :param low_fr: la fréquence de coupure inférieure (en réalité entier entre 0 et 1 si 0.1 on coupe les 10% première fr)
    :param high_frequency: la fréquence de coupure supérieure
    :param amplitude: l'amplitude du bruit
    :return: les données perturbées
    """
    for X_img in X_imgs:
        gaussian_noise_imgs = []
        row, col,ch = X_img.shape
        gauss = np.random.normal(0, 5, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        gauss = np.trunc(gauss)
        flatten_noise=gauss.flatten()
        noise_ft = np.fft.fft(flatten_noise) #calcule les fréquences de fourier de la série temporelle représentée par l'array
        #noise_ft=np.fft.ifft(np.fft.fft(gauss))
        low_frequency=int(len(noise_ft)*low_frequency)
        high_frequency=int(len(noise_ft)*high_frequency)
        for i in range(len(noise_ft)):
                if (i > high_frequency) or (i < low_frequency):
                        noise_ft[i] = 0
        noise_bp = np.real(np.fft.ifft(noise_ft)) # on a enlevé les fréquences extrèmes et on calcule la transformée in vers
        gauss=noise_bp.reshape((row, col,ch))
        X_img = X_img + gauss
        X_img[X_img < 0] = 0
        X_img[X_img > 255] = 255
        gaussian_noise_imgs.append(X_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.int16)
    return  gaussian_noise_imgs


def get_triggerset(trigger_params: dict):
    '''
    to be called by main.py or else to get the correct triggerset
    different generation methods may be used

    trigger_params:dict please refer to nomenclature.txt 
    '''
    # todo create all three methods of triggerset gen
    # you can use folder ./triggersets to store images
    n = trigger_params["n"]

    if trigger_params['from'] == "dataset":
        X_b, y_a = get_dataset(
            {'dataset': "cifar-10", "set": 'trigger', "n": 2*n, 'seed': trigger_params['seed']})

        # On enlève les images qui n'ont pas comme label y_a[0]
        X_a = []
        np.random.seed(trigger_params["seed"])
        label0 = y_a[np.random.randint(0, y_a.shape[0])]
        for i, label in enumerate(y_a):
            if label != label0:
                X_a.append(X_b[i])

        X_a = np.array(X_a)
        y_a = np.array(n*[label0])  # tous le même label

    else:
        # exterior set
        X_a = []
        for i in range(1, 101):
            img = plt.imread('triggersets/{}.jpg'.format(str(i)))
            X_a.append(np.array(img, dtype=np.uint8))
        X_a = np.array(X_a)
        labels = [0,1,2,3,4,5,6,7,8,9]
        np.random.seed(trigger_params["seed"])
        label0 = labels[np.random.randint(0, 11)]
        y_a = np.array(n*[label0])  # tous le même label

    # On réduit à n images
    if X_a.shape[0] >= n:
        X_a = X_a[:n]
    else:
        raise Exception("Not enough images")

    # On ajoute le bruit
    if trigger_params['noise']: ##== gaussian ou == bande passante
        sd = trigger_params['variance']
        X_a = add_gaussian_noise(X_a, sd)
    X_a = np.array(X_a, dtype=np.uint8)
    y_a = np.array(y_a, dtype=np.uint8)
    return(X_a, y_a)


if __name__ == "__main__":
    # test when coding this module
    X_a, y_a = get_triggerset({
        "n": 5,
        "variance": 5,
        "from": 'ext',
        "noise": False,
        "seed": 10
    })
    plt.imshow(X_a[0])
    plt.show()
