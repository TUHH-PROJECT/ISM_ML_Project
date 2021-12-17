import numpy as np
from skimage.measure import moments_hu,moments_central,moments_normalized

#Loading training images
X_train_gray = np.load("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray.npy")
#Loading verification images
X_ver_gray = np.load("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_ver_gray.npy")

#Initializing feature vectors for both datasets. Each image has 7 hu moments.
X_train_gray_hu = np.zeros((len(X_train_gray),7))
X_ver_gray_hu = np.zeros((len(X_ver_gray),7))

#Iterating through training images and extracting hu moments from them
for i in range(len(X_train_gray)):
    mu_train = moments_central(X_train_gray[i])
    nu_train = moments_normalized(mu_train)
    X_train_gray_hu[i] = moments_hu(nu_train)

#Iterating through verification images and extracting hu moments from them
for i in range(len(X_ver_gray)):
    mu_ver = moments_central(X_ver_gray[i])
    nu_ver = moments_normalized(mu_ver)
    X_ver_gray_hu[i] = moments_hu(nu_ver)

#Saving features
np.save("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/Features/X_train_gray_hu",X_train_gray_hu)
np.save("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/Features/Verification_Features/X_ver_gray_hu",X_ver_gray_hu)