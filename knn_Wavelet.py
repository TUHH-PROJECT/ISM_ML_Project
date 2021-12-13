import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# importation data of the features and labels
X_train_gray_wavelet_average = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_wavelet_average_combined.npy")
X_train_gray_wavelet_energy = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_wavelet_energy_combined.npy")
X_train_gray_wavelet_entropy = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_wavelet_entropy_combined.npy")
X_train_gray_wavelet_rms = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_wavelet_rms_combined.npy")
X_train_gray_wavelet_kurtosis = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_wavelet_kurtosis_combined.npy")
X_train_gray_wavelet_skewness = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_wavelet_skewness_combined.npy")
X_train_gray_wavelet_std = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_wavelet_std_combined.npy")

Y_train = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/y_train.npy")


X_ver_gray_wavelet_average = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_wavelet_average_combined.npy")
X_ver_gray_wavelet_energy = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_wavelet_energy_combined.npy")
X_ver_gray_wavelet_entropy = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_wavelet_entropy_combined.npy")
X_ver_gray_wavelet_rms = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_wavelet_rms_combined.npy")
X_ver_gray_wavelet_kurtosis = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_wavelet_kurtosis_combined.npy")
X_ver_gray_wavelet_skewness = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_wavelet_skewness_combined.npy")
X_ver_gray_wavelet_std = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_wavelet_std_combined.npy")

Y_ver = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/y_ver.npy")


# watch if the labels are uniformly samples

# c1,c2,c3,c4 = 0,0,0,0
# for x in range(0,len(Y_train)):
#     if (Y_train[x] == 0.):
#         c1+=1
#     elif (Y_train[x] == 1.):
#         c2+=1
#     elif (Y_train[x] == 2.):
#         c3+=1
#     elif (Y_train[x] == 3.):
#         c4+=1
#
# print("samples are ",c1,c2,c3,c4)
# c1,c2,c3,c4 = c1/len(Y_train),c2/len(Y_train),c3/len(Y_train),c4/len(Y_train)
# print("samples are ",c1,c2,c3,c4)

# create features matrix to train the model

X_train = np.array([X_train_gray_wavelet_average,X_train_gray_wavelet_energy, X_train_gray_wavelet_entropy, X_train_gray_wavelet_kurtosis,X_train_gray_wavelet_rms, X_train_gray_wavelet_skewness_gray_skewness, X_train_gray_wavelet_std])
X_train = np.transpose(X_train)

# create features matrix to test the model

X_ver = np.array([X_ver_gray_wavelet_average,X_ver_gray_wavelet_energy, X_ver_gray_wavelet_entropy, X_ver_gray_wavelet_kurtosis,X_ver_gray_wavelet_rms, X_ver_gray_wavelet_skewness_gray_skewness, X_ver_gray_wavelet_std])
X_ver = np.transpose(X_ver)

# Normalization of features
# Fit scaler on trainings data

# scaler = MinMaxScaler().fit(X_train)
scaler = RobustScaler().fit(X_train)
#Normalize trainings data
X_train = scaler.transform(X_train)
#Normalize verification data
X_ver = scaler.transform(X_ver)

# creating and training the model for k neighbors

weights = 'distance'
knn_model = KNeighborsClassifier(n_neighbors=20, weights = weights)
knn_model.fit(X_train,Y_train)

predicted = knn_model.predict(X_ver)
acc = accuracy_score(Y_ver,predicted)
print(acc)

# loop to search the best numbers of neighbour between a and b with step
a = 1
b = 200
step = 1
L1=[]
L2=[]

for k in range(a,b,step):
    knn_model = KNeighborsClassifier(n_neighbors=k, weights = weights)
    knn_model.fit(X_train,Y_train)
    predicted = knn_model.predict(X_ver)
    acc = accuracy_score(Y_ver, predicted)
    L1.append(k)
    L2.append(acc)


max_value = max(L2)
index_max_value = L2.index(max_value)
best_n_neighbour = L1[index_max_value]

print("best prediction is", max_value)
print("and it is for k =  ", best_n_neighbour)
plt.plot(L1, L2)
plt.show()

# calculate predicted with the best_n_neighbour

knn_model = KNeighborsClassifier(n_neighbors = best_n_neighbour, weights = weights)
knn_model.fit(X_train,Y_train)
predicted = knn_model.predict(X_ver)
acc = accuracy_score(Y_ver,predicted)
print(acc)

# #Score the model with the Classification Matrix from sklearn
# #For more information see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
labels = [0,1,2,3]
names = ['Normal','COVID','pneumonia','Lung_Opacity']
report = classification_report(Y_ver, predicted, labels=labels, target_names=names)
print(report)
with open('C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/ISM_ML_Project/Report_knn_Basics.txt','w',encoding='utf-8') as f:
    f.write(report)
