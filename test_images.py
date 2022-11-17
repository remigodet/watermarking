import matplotlib.pyplot as plt
import numpy as np
for i in range(1, 101):
    img = plt.imread('triggersets/ext_pics/{}.jpg'.format(str(i)))
    img = img[:32, :32]
    plt.imsave('triggersets/{}.jpg'.format(str(i)), img)
