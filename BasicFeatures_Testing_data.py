import numpy as np
import skimage.measure
from numpy import mean, sqrt, square
from scipy.stats import kurtosis, skew
import os
import pickle

testing_data = ['test/test_gray.npy','Mtec_test/mtec_test_gray.npy','Noisy_Test/test_noise_gray']
help = 0

for data in testing_data:
    #Data loading
    if help <= 1:
        image = np.load(os.path.join('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing',data))
    else:
        with open(os.path.join('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing',data), "rb") as f:
            image = pickle.load(f)
            f.close()
    
    #Calculating features
    features = np.zeros([len(image),6])

    for i in range(len(image)):
        features[i,0]=np.mean(image[i])
    
    for i in range(len(image)):
        features[i,1]=skimage.measure.shannon_entropy(image[i])
    
    for i in range(len(image)):
        features[i,2]=kurtosis(image[i], axis=None)

    for i in range(len(image)):
        features[i,3]=sqrt(mean(square(image[i])))

    for i in range(len(image)):
        features[i,4]=skew(image[i], axis=None)

    for i in range(len(image)):
        features[i,5]=np.std(image[i])

    print(features.shape)
    print(features[0:3,:])



    #Save the basic features
    if help == 0:
        np.save('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/BasicFeatures_testGray',features)
    if help == 1:
        np.save('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/BasicFeatures_mtecTestGray',features)
    if help == 2:
        np.save('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/BasicFeatures_noiseTestGray',features)

    help = help+1

