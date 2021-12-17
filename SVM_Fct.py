from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn import svm
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import os

def SVM_Classification(Feature1, save = False, report_name = 'NoName', model_name = 'NoName', Feature2 = 'None', Feature3 = 'None', Feature4 = 'None', Feature5 =  'None', Feature6 = 'None', Feature7 = 'None'):
    #Loading features and labels
    Features = [Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Feature7]
    nFeatures = 0
    if 'wavelet_avergae' in Features:
        nFeatures = nFeatures+1
        Feature_Xtrain = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_train_gray_wavelet_average_combined.npy")
        Feature_Xver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_ver_gray_wavelet_average_combined.npy")
        AllFeatures_XtrainGray = Feature_Xtrain
        AllFeatures_XverGray = Feature_Xver
    if 'wavelet_energy' in Features:
        nFeatures = nFeatures+1
        Feature_Xtrain = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_train_gray_wavelet_energy_combined.npy")
        Feature_Xver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_ver_gray_wavelet_energy_combined.npy")
        if nFeatures == 1:
            AllFeatures_XtrainGray = Feature_Xtrain
            AllFeatures_XverGray = Feature_Xver
        else:
            AllFeatures_XtrainGray = np.append(arr = AllFeatures_XtrainGray, values = Feature_Xtrain, axis = 1)
            AllFeatures_XverGray = np.append(arr = AllFeatures_XverGray, values = Feature_Xver, axis = 1)
    if 'wavelet_entropy' in Features:
        nFeatures = nFeatures+1
        Feature_Xtrain = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_train_gray_wavelet_entropy_combined.npy")
        Feature_Xver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_ver_gray_wavelet_entropy_combined.npy")
        if nFeatures == 1:
            AllFeatures_XtrainGray = Feature_Xtrain
            AllFeatures_XverGray = Feature_Xver
        else:
            AllFeatures_XtrainGray = np.append(arr = AllFeatures_XtrainGray, values = Feature_Xtrain, axis = 1)
            AllFeatures_XverGray = np.append(arr = AllFeatures_XverGray, values = Feature_Xver, axis = 1)
    if 'wavelet_kurtosis' in Features:
        nFeatures = nFeatures+1
        Feature_Xtrain = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_train_gray_wavelet_kurtosis_combined.npy")
        Feature_Xver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_ver_gray_wavelet_kurtosis_combined.npy")
        if nFeatures == 1:
            AllFeatures_XtrainGray = Feature_Xtrain
            AllFeatures_XverGray = Feature_Xver
        else:
            AllFeatures_XtrainGray = np.append(arr = AllFeatures_XtrainGray, values = Feature_Xtrain, axis = 1)
            AllFeatures_XverGray = np.append(arr = AllFeatures_XverGray, values = Feature_Xver, axis = 1)
    if 'wavelet_rms' in Features:
        nFeatures = nFeatures+1
        Feature_Xtrain = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_train_gray_wavelet_rms_combined.npy")
        Feature_Xver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_ver_gray_wavelet_rms_combined.npy")
        if nFeatures == 1:
            AllFeatures_XtrainGray = Feature_Xtrain
            AllFeatures_XverGray = Feature_Xver
        else:
            AllFeatures_XtrainGray = np.append(arr = AllFeatures_XtrainGray, values = Feature_Xtrain, axis = 1)
            AllFeatures_XverGray = np.append(arr = AllFeatures_XverGray, values = Feature_Xver, axis = 1)
    if 'wavelet_skewness' in Features:
        nFeatures = nFeatures+1
        Feature_Xtrain = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_train_gray_wavelet_skewness_combined.npy")
        Feature_Xver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_ver_gray_wavelet_skewness_combined.npy")
        if nFeatures == 1:
            AllFeatures_XtrainGray = Feature_Xtrain
            AllFeatures_XverGray = Feature_Xver
        else:
            AllFeatures_XtrainGray = np.append(arr = AllFeatures_XtrainGray, values = Feature_Xtrain, axis = 1)
            AllFeatures_XverGray = np.append(arr = AllFeatures_XverGray, values = Feature_Xver, axis = 1)
    if 'wavelet_std_deviation' in Features:
        nFeatures = nFeatures+1
        Feature_Xtrain = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_train_gray_wavelet_std_deviation_combined.npy")
        Feature_Xver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/WaveletFeatures_Single/X_ver_gray_wavelet_std_deviation_combined.npy")
        if nFeatures == 1:
            AllFeatures_XtrainGray = Feature_Xtrain
            AllFeatures_XverGray = Feature_Xver
        else:
            AllFeatures_XtrainGray = np.append(arr = AllFeatures_XtrainGray, values = Feature_Xtrain, axis = 1)
            AllFeatures_XverGray = np.append(arr = AllFeatures_XverGray, values = Feature_Xver, axis = 1)
    if 'Basic' in Features:
        nFeatures = nFeatures+1
        Feature_Xtrain = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/BasicFeatures_XtrainGray.npy")
        Feature_Xver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/BasicFeatures_XverGray.npy")
        if nFeatures == 1:
            AllFeatures_XtrainGray = Feature_Xtrain
            AllFeatures_XverGray = Feature_Xver
        else:
            AllFeatures_XtrainGray = np.append(arr = AllFeatures_XtrainGray, values = Feature_Xtrain, axis = 1)
            AllFeatures_XverGray = np.append(arr = AllFeatures_XverGray, values = Feature_Xver, axis = 1)
    if 'Shearlet' in Features:
        nFeatures = nFeatures+1
        Feature_Xtrain = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/shearlet_XtrainGray.npy")
        Feature_Xver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/shearlet_XverGray.npy")
        if nFeatures == 1:
            AllFeatures_XtrainGray = Feature_Xtrain
            AllFeatures_XverGray = Feature_Xver
        else:
            AllFeatures_XtrainGray = np.append(arr = AllFeatures_XtrainGray, values = Feature_Xtrain, axis = 1)
            AllFeatures_XverGray = np.append(arr = AllFeatures_XverGray, values = Feature_Xver, axis = 1)
    if 'Hu' in Features:
        nFeatures = nFeatures+1
        Feature_Xtrain = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_hu.npy")
        Feature_Xver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_ver_gray_hu.npy")
        if nFeatures == 1:
            AllFeatures_XtrainGray = Feature_Xtrain
            AllFeatures_XverGray = Feature_Xver
        else:
            AllFeatures_XtrainGray = np.append(arr = AllFeatures_XtrainGray, values = Feature_Xtrain, axis = 1)
            AllFeatures_XverGray = np.append(arr = AllFeatures_XverGray, values = Feature_Xver, axis = 1)

    #Loading the labels
    y_train = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Trainings_Vectors/Grey_Train_Verification_split/y_train.npy")
    y_ver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Trainings_Vectors/Grey_Train_Verification_split/y_ver.npy")

    #Normalization of features
    #Fit scaler on trainings data
    norm = RobustScaler().fit(AllFeatures_XtrainGray)
    #Normalize trainings data
    AllFeatures_XtrainGray_norm = norm.transform(AllFeatures_XtrainGray)
    #Normalize verification data
    AllFeatures_XverGray_norm = norm.transform(AllFeatures_XverGray)

    #Create the linear SVM model
    #For more information see: https://scikit-learn.org/stable/modules/svm.html
    SVM= svm.SVC()
    SVM.fit(AllFeatures_XtrainGray_norm, y_train)

    #Prediction
    y_pred = SVM.predict(AllFeatures_XverGray_norm)

    #Accuracy
    acc = accuracy_score(y_ver, y_pred)

    if save == True:
        #Score the model with the Classification Matrix from sklearn
        #For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
        labels = [0,1,2,3]
        names = ['Normal','COVID','pneumonia','Lung_Opacity']
        report = classification_report(y_ver, y_pred, labels=labels, target_names=names)
        #Save the Report
        with open(os.path.join('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/SVM', report_name), 'w', encoding='utf-8') as f:
            f.write(report)
        #Save the model: Reload it with: SVM= joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/SVM/Model_name.sav")
        joblib.dump(SVM, os.path.join("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/SVM", model_name))
    return acc
