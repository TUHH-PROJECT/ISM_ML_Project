import numpy as np
import skimage.measure
from numpy import mean, sqrt, square
from scipy.stats import kurtosis, skew

#Loading the verification dataset images
X_train_gray = np.load("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_train_gray.npy")

#Average
average = np.zeros(len(X_train_gray))

for i in range(len(X_train_gray)):
    average[i]=np.mean(X_train_gray[i])

np.save("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/Features/X_train_gray_average",average)

#Entropy
entropy = np.zeros(len(X_train_gray))

for i in range(len(X_train_gray)):
    entropy[i] = skimage.measure.shannon_entropy(X_train_gray[i])

np.save("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/Features/X_train_gray_entropy",entropy)

#Kurtosis and Skewness
X_train_gray_kurtosis = np.zeros(len(X_train_gray))
skewness = np.zeros(len(X_train_gray))

for i in range(len(X_train_gray)):
    X_train_gray_kurtosis[i] = kurtosis(X_train_gray[i], axis=None)
    skewness[i] = skew(X_train_gray[i], axis=None)

np.save("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/Features/X_train_gray_kurtosis.npy",X_train_gray_kurtosis)
np.save("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/Features/X_train_gray_skewness",skewness)

#RMS
rms = np.zeros(len(X_train_gray))
X_train_gray_squared = square(X_train_gray)

for i in range(len(X_train_gray)):
    rms[i]=sqrt(mean(X_train_gray_squared[i]))

np.save("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/Features/X_train_gray_rms",rms)

#Standard Deviation
std_deviation = np.zeros(len(X_train_gray))

for i in range(len(X_train_gray)):
    std_deviation[i] = np.std(X_train_gray[i])

np.save("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/Features/X_train_gray_std",std_deviation)

#Energy
energy = np.zeros(len(X_train_gray))

for i in range(len(X_train_gray)):
    energy[i] = np.sum(square(X_train_gray[i]))

np.save("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/Features/X_train_gray_energy",energy)