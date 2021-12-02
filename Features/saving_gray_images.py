from skimage import io
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import os


X_train = np.load("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train.npy")

X_train_gray = []

for image in X_train:
    X_train_gray.append(rgb2gray(image))

X_train_gray = np.array(X_train_gray)
print(X_train_gray.shape)
np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray",X_train_gray)

