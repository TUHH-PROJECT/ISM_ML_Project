from skimage.color import rgb2gray
import numpy as np
import pickle

#Converting the test data into grayscale
test = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Test/test.npy")
print(test.shape)
test_gray = []

for image in test:
    test_gray.append(rgb2gray(image))

test_gray = np.array(test_gray)
print(test_gray.shape)
np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Test/test_gray.npy",test_gray)

#Converting the mtec_test data into grayscale
mtec_test = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Mtec_test/mtec_test.npy")
print(mtec_test.shape)
mtec_test_gray = []

for image in mtec_test:
    mtec_test_gray.append(rgb2gray(image))

mtec_test_gray = np.array(mtec_test_gray)
print(mtec_test_gray.shape)
np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Mtec_test/mtec_test_gray.npy",mtec_test_gray)

#Converting the noisy test data into grayscale
with open("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Noisy_Test/test_noise", "rb") as f:
  test_noise = pickle.load(f)
f.close()
print(len(test_noise))

test_noise_gray = []

for image in test_noise:
    test_noise_gray.append(rgb2gray(image))

print(len(test_noise_gray))
print(test_noise_gray[0].shape)
with open("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Noisy_Test/test_noise_gray", "wb") as f:
  pickle.dump(test_noise_gray, f)
f.close()
