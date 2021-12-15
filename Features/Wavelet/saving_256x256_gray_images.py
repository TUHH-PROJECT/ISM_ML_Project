import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray

X_train_gray = np.load("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train.npy")
X_train_gray_resized = np.zeros((len(X_train_gray),256,256))

for i in range(len(X_train_gray_resized)):
    resized_image = resize(X_train_gray[i],(256,256))
    gray_image = rgb2gray(resized_image)
    X_train_gray_resized[i] = gray_image

np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray_resized",X_train_gray_resized)