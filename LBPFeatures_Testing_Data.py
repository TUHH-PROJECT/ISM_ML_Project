import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
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
    numPoints=8
    radius=3
    METHOD = 'uniform'

    features = []

    for im in image:
        lbp=feature.local_binary_pattern(im, numPoints, radius, method= METHOD)
        hist_ref, _= np.histogram(lbp, bins=2**numPoints, range=(0, 2**numPoints))
        features.append(hist_ref)

    features = np.array(features)

    print(features.shape)
    print(features[0:3,:])



    #Save the basic features
    if help == 0:
        np.save('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_testGray',features)
    if help == 1:
        np.save('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_mtecTestGray',features)
    if help == 2:
        np.save('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_noiseTestGray',features)

    help = help+1

