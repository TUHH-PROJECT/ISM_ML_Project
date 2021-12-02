import numpy as np
from numpy import mean, sqrt, square
from skimage import io, img_as_float
from skimage.transform import rescale, resize
from matplotlib import pyplot as plt
import cv2
from skimage.morphology import disk
from skimage.color import rgb2gray

X_train_gray = np.load("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray.npy")

rms = np.zeros(len(X_train_gray))
X_train_gray_squared = square(X_train_gray)

for i in range(len(X_train_gray)):
    rms[i]=sqrt(mean(X_train_gray_squared[i]))

print(rms)
print(len(rms))
print(rms[1])
print(sqrt(mean(square(X_train_gray[1]))))

np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray_rms",rms)