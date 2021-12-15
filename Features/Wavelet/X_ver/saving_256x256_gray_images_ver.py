import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray

X_ver_gray = np.load("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_ver.npy")
X_ver_gray_resized = np.zeros((len(X_ver_gray),256,256))

for i in range(len(X_ver_gray_resized)):
    resized_image = resize(X_ver_gray[i],(256,256))
    gray_image = rgb2gray(resized_image)
    X_ver_gray_resized[i] = gray_image

np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_ver_gray_resized",X_ver_gray_resized)