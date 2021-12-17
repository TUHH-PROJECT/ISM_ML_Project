import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray

#Loading training images
X_train_gray = np.load("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train.npy")

#Initializing image vector
X_train_gray_resized = np.zeros((len(X_train_gray),256,256))

#Iterating through training images while resizing them and changing them to grayscale images
for i in range(len(X_train_gray_resized)):
    resized_image = resize(X_train_gray[i],(256,256))
    gray_image = rgb2gray(resized_image)
    X_train_gray_resized[i] = gray_image

#Saving resized images
np.save("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/Features/Wavelet/X_train_gray_resized.npy",X_train_gray_resized)