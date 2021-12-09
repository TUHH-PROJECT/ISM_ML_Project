from skimage import io
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import os


X_train = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Trainings_Vectors/Train_verification_split/X_train.npy")
X_ver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Trainings_Vectors/Train_verification_split/X_ver.npy")

X_train_gray = []
X_ver_gray = []

for image in X_train:
    X_train_gray.append(rgb2gray(image))

for image in X_ver:
    X_ver_gray.append(rgb2gray(image))

X_train_gray = np.array(X_train_gray)
X_ver_gray = np.array(X_ver_gray)

print(X_train_gray.shape)
print(X_ver_gray.shape)

np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Trainings_Vectors/Grey_Train_Verification_split/X_train_gray",X_train_gray)
np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Trainings_Vectors/Grey_Train_Verification_split/X_ver_gray",X_ver_gray)

