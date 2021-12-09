import numpy as np

#Load all feature vectors
X_train_gray_average = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_average.npy")
X_train_gray_entropy = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_entropy.npy")
X_train_gray_kurtosis = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_kurtosis.npy")
X_train_gray_rms = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_rms.npy")
X_train_gray_skewness = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_skewness.npy")
X_train_gray_std = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_std.npy")

X_ver_gray_average = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_average.npy")
X_ver_gray_entropy = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_entropy.npy")
X_ver_gray_kurtosis = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_kurtosis.npy")
X_ver_gray_rms = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_rms.npy")
X_ver_gray_skewness = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_skewness.npy")
X_ver_gray_std = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_std.npy")

#Writing all feature vectors for X_train_gray and for X_ver_gray in one numpy matrix (nsample,nfeatures)
BasicFeatures_XtrainGray = np.transpose(np.array([X_train_gray_average, X_train_gray_entropy, X_train_gray_kurtosis, X_train_gray_rms, X_train_gray_skewness, X_train_gray_std ]))
BasicFeatures_XverGray = np.transpose(np.array([X_ver_gray_average, X_ver_gray_entropy, X_ver_gray_kurtosis, X_ver_gray_rms, X_ver_gray_skewness, X_ver_gray_std ]))

print(BasicFeatures_XtrainGray.shape)
print(BasicFeatures_XverGray.shape)

#Saving
np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/BasicFeatures_XtrainGray.npy",BasicFeatures_XtrainGray)
np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/BasicFeatures_XverGray.npy",BasicFeatures_XverGray)