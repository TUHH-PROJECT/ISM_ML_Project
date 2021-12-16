import numpy as np
from numpy import mean, sqrt, square
from skimage import io, img_as_float
from skimage.transform import rescale, resize
from matplotlib import pyplot as plt
import cv2
from skimage.morphology import disk
from skimage.color import rgb2gray

X_train_gray = np.load("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray.npy")

std_deviation = np.zeros(len(X_train_gray))

for i in range(len(X_train_gray)):
    std_deviation[i] = np.std(X_train_gray[i])

print(std_deviation)
print(len(std_deviation))
print(std_deviation[1])
print(np.std(X_train_gray[1]))

np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray_std",std_deviation)