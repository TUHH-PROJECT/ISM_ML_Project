import skimage.measure
import numpy as np
from numpy import mean, sqrt, square
from skimage import io, img_as_float
from skimage.transform import rescale, resize
from matplotlib import pyplot as plt
import cv2
from skimage.morphology import disk
from skimage.color import rgb2gray
from scipy.stats import kurtosis, skew

X_train_gray = np.load("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray.npy")

X_train_gray_entropy = np.zeros(len(X_train_gray))

for i in range(len(X_train_gray)):
    X_train_gray_entropy[i] = skimage.measure.shannon_entropy(X_train_gray[i])

print(X_train_gray_entropy)
print(len(X_train_gray_entropy))
print(X_train_gray_entropy[1])
print(skimage.measure.shannon_entropy(X_train_gray[1]))

np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray_entropy",X_train_gray_entropy)

