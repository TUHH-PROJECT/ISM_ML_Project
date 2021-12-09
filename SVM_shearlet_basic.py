from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import svm
import joblib
from sklearn.metrics import classification_report

#Loading features and labels
BasicFeatures_XtrainGray = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/BasicFeatures_XtrainGray.npy")
BasicFeatures_XverGray = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/BasicFeatures_XverGray.npy")
shearlet_XtrainGray = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/shearlet_XtrainGray.npy")
shearlet_XverGray = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Feature Vectors/shearlet_XverGray.npy")
y_train = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Trainings_Vectors/Grey_Train_Verification_split/y_train.npy")
y_ver = np.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/Trainings_Vectors/Grey_Train_Verification_split/y_ver.npy")

Basic_Shearlet_XtrainGray = np.append(arr=BasicFeatures_XtrainGray,values=shearlet_XtrainGray,axis=1)
Basic_Shearlet_XverGray = np.append(arr=BasicFeatures_XverGray,values=shearlet_XverGray,axis=1)

#Normalization of features
#Fit scaler on trainings data
norm = MinMaxScaler().fit(Basic_Shearlet_XtrainGray)
#Normalize trainings data
XtrainGray_norm = norm.transform(Basic_Shearlet_XtrainGray)
#Normalize verification data
XverGray_norm = norm.transform(Basic_Shearlet_XverGray)

#Create the SVM model
#For more information see: https://scikit-learn.org/stable/modules/svm.html
SVM = svm.SVC()
SVM.fit(XtrainGray_norm, y_train)

#Save the model: Reload it with: SVM_Basic = joblib.load("/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/SVM/SVM_Basic.sav")
joblib.dump(SVM, "/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/SVM/SVM_Basic_Shearlet.sav")

#Prediction
y_pred = SVM.predict(XverGray_norm)

#Score the model with the Classification Matrix from sklearn
#For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
labels = [0,1,2,3]
names = ['Normal','COVID','pneumonia','Lung_Opacity']
report = classification_report(y_ver, y_pred, labels=labels, target_names=names)
print(report)
with open('/Users/giulianotaccogna/Documents/Development/Python/ISM_ML_Project/Data/ML_Models/SVM/Report_SVM_Basic_Shearlet.txt', 'w', encoding='utf-8') as f:
    f.write(report)

