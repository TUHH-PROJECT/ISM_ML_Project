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

X_train_gray_kurtosis = np.zeros(len(X_train_gray))
X_train_gray_skewness = np.zeros(len(X_train_gray))

for i in range(len(X_train_gray)):
    X_train_gray_kurtosis[i] = kurtosis(X_train_gray[i], axis=None)
    X_train_gray_skewness[i] = skew(X_train_gray[i], axis=None)

print(X_train_gray_kurtosis)
print(len(X_train_gray_kurtosis))
print(X_train_gray_kurtosis[1])
print(kurtosis(X_train_gray[1], axis=None))

print(X_train_gray_skewness)
print(len(X_train_gray_skewness))
print(X_train_gray_skewness[1])
print(skew(X_train_gray[1], axis=None))

np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray_kurtosis",X_train_gray_kurtosis)
np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray_skewness",X_train_gray_skewness)