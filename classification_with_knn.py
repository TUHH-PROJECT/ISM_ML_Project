import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# importation des données
features_1 = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_average.npy")
features_2 = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_entropy.npy")
features_3 = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_rms.npy")
features_4 = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_kurtosis.npy")
features_5 = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_skewness.npy")
features_6 = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_std.npy")

labels = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/y_train.npy")

ver_1 = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_average.npy")
ver_2 = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_entropy.npy")
ver_3 = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_rms.npy")
ver_4 = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_kurtosis.npy")
ver_5 = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_skewness.npy")
ver_6 = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_std.npy")

y_ver = np.load(
    "C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/y_ver.npy")


# Normalization of the features

def normalize(X, a, b):
    Xmin = min(X)
    Xmax = max(X)
    for k in range(0, len(X)):
        X[k] = a + (X[k] - Xmin) / (Xmax - Xmin) * (b - a)


lower_bound = -100
upper_bound = +100

normalize(features_1, lower_bound, upper_bound)
normalize(features_2, lower_bound, upper_bound)
normalize(features_3, lower_bound, upper_bound)
normalize(features_4, lower_bound, upper_bound)
normalize(features_5, lower_bound, upper_bound)
normalize(features_6, lower_bound, upper_bound)
normalize(ver_1, lower_bound, upper_bound)
normalize(ver_2, lower_bound, upper_bound)
normalize(ver_3, lower_bound, upper_bound)
normalize(ver_4, lower_bound, upper_bound)
normalize(ver_5, lower_bound, upper_bound)
normalize(ver_6, lower_bound, upper_bound)


# create features matrix to train the model

features = np.array([features_1, features_2, features_3,
                    features_4, features_5, features_6])
# features = np.array([features_2,features_3,features_5,features_6])
features = np.transpose(features)

# create features matrix to test the model

ver = np.array([ver_1, ver_2, ver_3, ver_4, ver_5, ver_6])
# ver = np.array([ver_2,ver_3,ver_5,ver_6])
ver = np.transpose(ver)


# creating and training the model

# knn_model = KNeighborsClassifier(n_neighbors=2000,weights = 'distance')
# knn_model.fit(features,labels)
#
# # prediction f the model
# predicted = knn_model.predict(ver)
# acc = accuracy_score(y_ver,predicted)
# print(acc)

# loop to search the best numbers of neighbour

L1 = []
L2 = []
for k in range(1, 500, 5):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(features, labels)
    predicted = knn_model.predict(ver)
    acc = accuracy_score(y_ver, predicted)
    L1.append(k)
    L2.append(acc)
max_value = max(L2)
max_index = L2.index(max_value)
best_n_neighbour = L1[max_index]

print(max_value)
print("best_neighbour is ", best_n_neighbour)
plt.plot(L1, L2)
plt.show()
