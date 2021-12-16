# TODO add documentation
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

x_train = np.load("/Users/apoorv/PycharmProjects/ISM_ML_Project/Features/wavelet/training vectors/X_train_gray_wavelet_rms_combined.npy")
x_test = np.load("/Users/apoorv/PycharmProjects/ISM_ML_Project/Features/wavelet/test vectors/X_ver_gray_wavelet_rms_combined.npy")

y_train = np.load("/Users/apoorv/PycharmProjects/ISM_ML_Project/labels/y_train.npy")
y_test = np.load("/Users/apoorv/PycharmProjects/ISM_ML_Project/labels/y_test.npy")

print("------------------------------ normalizing feature vectors ----------------------------------------------------\n")
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("------------------------------ creating model -----------------------------------------------------------------\n")
model = LogisticRegression(solver= 'newton-cg', C = 1.0, multi_class= 'multinomial', random_state= 0)
model.fit(x_train, y_train)
print("------------------------------ model created ------------------------------------------------------------------\n")

y = model.predict(x_test)

print("------------------------------ Result -------------------------------------------------------------------------\n")

print("******** Acccuracy score for training set: ")
print(model.score(x_train, y_train))
print("************ Acccuracy score for test set: ")
print(model.score(x_test, y_test))

report = classification_report(y_test, y, labels=[0, 1, 2, 3], target_names=['Normal','COVID','pneumonia','Lung_Opacity'])
print(report)


