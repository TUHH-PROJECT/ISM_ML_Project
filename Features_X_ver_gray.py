import numpy as np
from skimage import io, img_as_float
from skimage.transform import rescale, resize
from matplotlib import pyplot as plt
from skimage.morphology import disk
from skimage.color import rgb2gray
import skimage.measure
from numpy import mean, sqrt, square
from scipy.stats import kurtosis, skew

#Average
X_ver_gray = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Trainings_Vectors/Grey_Train_Verification_split/X_ver_gray.npy")

average = np.zeros(len(X_ver_gray))

for i in range(len(X_ver_gray)):
    average[i]=np.mean(X_ver_gray[i])

print(average)
print(len(average))
print(average[0])
print(np.mean(X_ver_gray[0]))

np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_average",average)

#Entropy
X_ver_gray_entropy = np.zeros(len(X_ver_gray))

for i in range(len(X_ver_gray)):
    X_ver_gray_entropy[i] = skimage.measure.shannon_entropy(X_ver_gray[i])

print(X_ver_gray_entropy)
print(len(X_ver_gray_entropy))
print(X_ver_gray_entropy[1])
print(skimage.measure.shannon_entropy(X_ver_gray[1]))

np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_entropy",X_ver_gray_entropy)

#Kurtosis and Skewness
X_ver_gray_kurtosis = np.zeros(len(X_ver_gray))
X_ver_gray_skewness = np.zeros(len(X_ver_gray))

for i in range(len(X_ver_gray)):
    X_ver_gray_kurtosis[i] = kurtosis(X_ver_gray[i], axis=None)
    X_ver_gray_skewness[i] = skew(X_ver_gray[i], axis=None)

print(X_ver_gray_kurtosis)
print(len(X_ver_gray_kurtosis))
print(X_ver_gray_kurtosis[1])
print(kurtosis(X_ver_gray[1], axis=None))

print(X_ver_gray_skewness)
print(len(X_ver_gray_skewness))
print(X_ver_gray_skewness[1])
print(skew(X_ver_gray[1], axis=None))

np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_kurtosis.npy",X_ver_gray_kurtosis)
np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_skewness",X_ver_gray_skewness)

#RMS
rms = np.zeros(len(X_ver_gray))
X_ver_gray_squared = square(X_ver_gray)

for i in range(len(X_ver_gray)):
    rms[i]=sqrt(mean(X_ver_gray_squared[i]))

print(rms)
print(len(rms))
print(rms[1])
print(sqrt(mean(square(X_ver_gray[1]))))

np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_rms",rms)

#Standard Deviation
std_deviation = np.zeros(len(X_ver_gray))

for i in range(len(X_ver_gray)):
    std_deviation[i] = np.std(X_ver_gray[i])

print(std_deviation)
print(len(std_deviation))
print(std_deviation[1])
print(np.std(X_ver_gray[1]))

np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_std",std_deviation)