import numpy as np
from skimage import io, img_as_float
from skimage.transform import rescale, resize
from matplotlib import pyplot as plt
import cv2
from skimage.morphology import disk
from skimage.color import rgb2gray

X_train_gray = np.load("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray.npy")

average = np.zeros(len(X_train_gray))

for i in range(len(X_train_gray)):
    average[i]=np.mean(X_train_gray[i])

print(average)
print(len(average))
print(average[0])
print(np.mean(X_train_gray[0]))

np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray_average",average)
