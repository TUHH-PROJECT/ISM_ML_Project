from skimage import feature
import matplotlib.pyplot as plt
import numpy as np

X_train_gray = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray.npy")

X_ver_gray = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray.npy")

print(X_train_gray.shape)

numPoints=8
radius=3
METHOD = 'uniform'

X_train_gray_lbp_hist = []
X_ver_gray_lbp_hist = []

for image in X_train_gray:
    lbp=feature.local_binary_pattern(image, numPoints, radius, method= METHOD)
    hist_ref, _= np.histogram(lbp, bins=2**numPoints, range=(0, 2**numPoints))
    X_train_gray_lbp_hist.append(hist_ref)

for image in X_ver_gray:
    lbp=feature.local_binary_pattern(image, numPoints, radius, method= METHOD)
    hist_ref, _= np.histogram(lbp, bins=2**numPoints, range=(0, 2**numPoints))
    X_ver_gray_lbp_hist.append(hist_ref)

X_train_gray_lbp_hist = np.array(X_train_gray_lbp_hist)
X_ver_gray_lbp_hist = np.array(X_ver_gray_lbp_hist)

np.save("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_lbp_hist.npy",X_train_gray_lbp_hist)
np.save("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_lbp_hist.npy",X_ver_gray_lbp_hist)
