from sklearn.naive_bayes import GaussianNB
import numpy as np
import joblib

#Loading Training Feature
lbp = np.load("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/Features/X_train_gray_lbp_hist.npy")

#Loading Verification Feature
lbp_ver = np.load("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/Features/Verification_Features/X_ver_gray_lbp_hist.npy")

#Loading training and test verification sets
dataset_train = lbp
dataset_ver = lbp_ver
labels_train = np.load("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/y_train.npy")
labels_ver = np.load("C:/Users/ASUS/Documents/GitHub/ISM_ML_Project/y_ver.npy")

#Gaussian
x_train, x_test, y_train, y_test = dataset_train, dataset_ver, labels_train, labels_ver
gnb = GaussianNB()
model = gnb.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = model.score(x_test,y_test)
print("Accuracy with Gaussian distribution is: " + str(accuracy))
joblib.dump(model,"NaiveBayesModel")
