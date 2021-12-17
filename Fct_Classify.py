import numpy as np
import os
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

def classify (Model, Dataset):
    if Dataset == 'Test':
        with open("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Test/test_names", "rb") as f:
            names = pickle.load(f)
        f.close()
        if Model == 'KNN_LBP':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/knn_model_LBP")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_testGray.npy')
            train_data = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_lbp_hist.npy')
            norm = RobustScaler().fit(train_data)
        if Model == 'KNN_Basic':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/knn_model_Basics")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/BasicFeatures_testGray.npy')
            train_data = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/BasicFeatures_XtrainGray.npy')
            norm = RobustScaler().fit(train_data)
        if Model == 'SVM_LBP':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/LBP_SVM_RobustScaler.sav")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_testGray.npy')
            train_data = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_lbp_hist.npy')
            norm = RobustScaler().fit(train_data)
        if Model == 'LR_LBP':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/lr_model.sav")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_testGray.npy')
            train_data = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_lbp_hist.npy')
            norm = StandardScaler().fit(train_data)
        if Model == 'NB_LBP':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/NaiveBayesModel.sav")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_testGray.npy')
    
    
    if Dataset == 'Noisy_Test':
        with open("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Noisy_Test/test_noise_names", "rb") as f:
            names = pickle.load(f)
        f.close()
        if Model == 'KNN_LBP':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/knn_model_LBP")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_noiseTestGray.npy')
            train_data = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_lbp_hist.npy')
            norm = RobustScaler().fit(train_data)
        if Model == 'KNN_Basic':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/knn_model_Basics")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/BasicFeatures_noiseTestGray.npy')
            train_data = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/BasicFeatures_XtrainGray.npy')
            norm = RobustScaler().fit(train_data)
        if Model == 'SVM_LBP':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/LBP_SVM_RobustScaler.sav")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_noiseTestGray.npy')
            train_data = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_lbp_hist.npy')
            norm = RobustScaler().fit(train_data)
        if Model == 'LR_LBP':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/lr_model.sav")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_noiseTestGray.npy')
            train_data = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_lbp_hist.npy')
            norm = StandardScaler().fit(train_data)
        if Model == 'NB_LBP':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/NaiveBayesModel.sav")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_noiseTestGray.npy')
            
            

    if Dataset == 'Mtec':
        with open("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Mtec_test/mtec_test_names", "rb") as f:
            names = pickle.load(f)
        f.close()
        if Model == 'KNN_LBP':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/knn_model_LBP")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_mtecTestGray.npy')
            train_data = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_lbp_hist.npy')
            norm = RobustScaler().fit(train_data)
        if Model == 'KNN_Basic':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/knn_model_Basics")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/BasicFeatures_mtecTestGray.npy')
            train_data = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/BasicFeatures_XtrainGray.npy')
            norm = RobustScaler().fit(train_data)
        if Model == 'SVM_LBP':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/LBP_SVM_RobustScaler.sav")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_mtecTestGray.npy')
            train_data = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_lbp_hist.npy')
            norm = RobustScaler().fit(train_data)
        if Model == 'LR_LBP':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/lr_model.sav")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_mtecTestGray.npy')
            train_data = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/X_train_gray_lbp_hist.npy')
            norm = StandardScaler().fit(train_data)
        if Model == 'NB_LBP':
            classi = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/NaiveBayesModel.sav")
            features = np.load('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Features/LBP_mtecTestGray.npy')
        
            
    
    #Normalization
    if Model == 'NB_LBP':
        features_norm = features
    else:
        features_norm = norm.transform(features)

    #Prediction
    y_pred = classi.predict(features_norm)

    #Translating the integers in y_pred in the actual labels
    classes = [None]*len(y_pred)
    i = 0
    for int in y_pred:
        if int == 0:
            classes[i] = 'Normal'
        if int == 1:
            classes[i] = 'COVID'
        if int == 2:
            classes[i] = 'pneumonia'
        if int == 3:
            classes[i] = 'Lung_Opacity'
        i = i+1

    #Writing the result in a text file
    f = open(os.path.join('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Testing/Classifications', Dataset+'_'+Model+'.txt'), 'w', encoding='utf-8')
    for i in range(len(names)):
        f.write(names[i]+' '+classes[i]+'\n')
    f.close()


Datasets = ['Test', 'Noisy_Test', 'Mtec']
Models = ['KNN_LBP', 'KNN_Basic', 'SVM_LBP', 'LR_LBP', 'NB_LBP']

for dataset in Datasets:
    for model in Models:
        classify(model, dataset)
    




  
   


