import numpy as np
import pywt
import matplotlib.pyplot as plt
import skimage.measure
from scipy.stats import kurtosis, skew
from numpy import mean, sqrt, square

X_ver_gray = np.load("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/X_ver_gray_resized.npy")

average = np.zeros((len(X_ver_gray),3))
entropy = np.zeros((len(X_ver_gray),3))
kurtosis_train = np.zeros((len(X_ver_gray),3))
skewness = np.zeros((len(X_ver_gray),3))
rms = np.zeros((len(X_ver_gray),3))
#X_train_gray_squared = square(X_ver_gray)
std_deviation = np.zeros((len(X_ver_gray),3))
energy = np.zeros((len(X_ver_gray),3))

n = 1
w = 'db1'

for i in range(len(X_ver_gray)):
    coeffs = pywt.wavedec2(X_ver_gray[i], wavelet=w, level=n)
    # normalize each coefficient array
    coeffs[0] /= np.abs(coeffs[0]).max()
    for detail_level in range(n):
        coeffs[detail_level + 1] = [d/np.abs(d).max() for d in coeffs[detail_level + 1]]
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    for j in range(3):
        if j == 0:
            part = 'da'
        elif j == 1:
            part = 'ad'
        else:
            part = 'dd'
        average[i][j] = np.mean(arr[coeff_slices[1][part]])
        entropy[i][j] = skimage.measure.shannon_entropy(arr[coeff_slices[1][part]])
        kurtosis_train[i][j] = kurtosis(arr[coeff_slices[1][part]], axis=None)
        skewness[i][j] = skew(arr[coeff_slices[1][part]], axis=None)
        rms[i][j] = sqrt(mean(square(arr[coeff_slices[1][part]])))
        std_deviation[i][j] = np.std(arr[coeff_slices[1][part]])
        energy[i][j] = np.sum(arr[coeff_slices[1][part]])//(arr[coeff_slices[1][part]].shape[0]*arr[coeff_slices[1][part]].shape[1])

np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/Wavelet/X_ver/X_train_gray_wavelet_average_combined",average)
np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/Wavelet/X_ver/X_train_gray_wavelet_entropy_combined",entropy)
np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/Wavelet/X_ver/X_train_gray_wavelet_kurtosis_combined",kurtosis_train)
np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/Wavelet/X_ver/X_train_gray_wavelet_skewness_combined",skewness)
np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/Wavelet/X_ver/X_train_gray_wavelet_rms_combined",rms)
np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/Wavelet/X_ver/X_train_gray_wavelet_std_deviation_combined", std_deviation)
np.save("C:/Users/ASUS/Desktop/ISM/python_test/Training_Verification_Split/Wavelet/X_ver/X_train_gray_wavelet_energy_combined",energy)