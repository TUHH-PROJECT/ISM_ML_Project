import numpy as np
from scipy import ndimage as img
from scipy import io as sio
import matplotlib.pyplot as plt
import pyshearlab
import Entropy_and_Energy_Fct

#Import X_train_gray and X_ver-gray
X_train_gray = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Trainings_Vectors/Grey_Train_Verification_split/X_train_gray.npy")
X_ver_gray = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Trainings_Vectors/Grey_Train_Verification_split/X_ver_gray.npy")

#Performing the Shearlet Transform for all Images in X_train_gray and X_ver_gray and calculating
#the energy and the entropy of each transformed image
nFeatures_per_Image = 98
#Preallocating a empty matrix to save the features for each image
shearlet_XtrainGray = np.zeros([X_train_gray.shape[0], nFeatures_per_Image])
shearlet_XverGray = np.zeros([X_ver_gray.shape[0], nFeatures_per_Image])

#Create shearlets. For a detail explanation of the function see: http://na.math.uni-goettingen.de/pyshearlab/pyShearLab2D.m.html
scales = 4
#SL is a array, specifying the level of shearing occuring on each scale
SL = np.array([1, 1, 2, 2])
shearletSystem = pyshearlab.SLgetShearletSystem2D(0,X_ver_gray.shape[1], X_ver_gray.shape[2], scales, SL)
print("Number of Shearlets: " + str(shearletSystem['nShearlets']))

for i in range(X_train_gray.shape[0]):
  #Performing the Decompostion
  coeffs = pyshearlab.SLsheardec2D(X_train_gray[i,:,:], shearletSystem)
  #Counter
  if i%500 == 0:
      print(i)
  #Calculating the energy and entropy from each shearlet
  for j in range(coeffs.shape[2]):
    shearlet_XtrainGray[i,2*j]= Entropy_and_Energy_Fct.entropy_shannon(coeffs[:,:,j])
    shearlet_XtrainGray[i,2*j+1] = Entropy_and_Energy_Fct.normalized_energy_2d(coeffs[:,:,j])

for i in range(X_ver_gray.shape[0]):
  #Performing the Decompostion
  coeffs = pyshearlab.SLsheardec2D(X_ver_gray[i,:,:], shearletSystem)
  #Counter
  if i%500 == 0:
     print(i)
  #Calculating the energy and entropy from each shearlet
  for j in range(coeffs.shape[2]):
    shearlet_XverGray[i,2*j]= Entropy_and_Energy_Fct.entropy_shannon(coeffs[:,:,j])
    shearlet_XverGray[i,2*j+1] = Entropy_and_Energy_Fct.normalized_energy_2d(coeffs[:,:,j])

np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/shearlet_XtrainGray",shearlet_XtrainGray)
np.save("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/shearlet_XverGray",shearlet_XverGray)