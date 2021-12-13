import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#TODO: change comment to english please
# importation des données
features_1 = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_average.npy")
features_2 = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_entropy.npy")
features_3 = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_rms.npy")
features_4 = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_kurtosis.npy")
features_5 = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_skewness.npy")
features_6 = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_train_gray_std.npy")

labels = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/y_train.npy")

ver_1 = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_average.npy")
ver_2 = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_entropy.npy")
ver_3 = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_rms.npy")
ver_4 = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_kurtosis.npy")
ver_5 = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_skewness.npy")
ver_6 = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/X_ver_gray_std.npy")

y_ver = np.load("C:/Users/jemho/Documents/Etude/hambourg/cours_hambourg/intelligent system in medicin/pbl/Feature Vectors/y_ver.npy")

#TODO change comment to english
# création de la matrice qui entraïne le model

# features = np.array([features_1,features_2,features_3,features_4,features_5,features_6])
features = np.array([features_2,features_3,features_5,features_6])
features = np.transpose(features)

#TODO change comment to english
# création de la matrice pur testé le model

# ver = np.array([ver_1,ver_2,ver_3,ver_4,ver_5,ver_6])
ver = np.array([ver_2,ver_3,ver_5,ver_6])
ver = np.transpose(ver)

#TODO change comment to english
# création et entrainement du model

# knn_model = KNeighborsClassifier(n_neighbors=20)
# knn_model.fit(features,labels)
#
# # prediction du model
# predicted = knn_model.predict(ver)
# acc = accuracy_score(y_ver,predicted)
# print(acc)


# ce que fait le accuracy_score
# compteur = 0
#
# for k in range(1,len(ver_1)):
#     if (predicted[k] == y_ver[k]):
#         compteur += 1
# print(compteur/3386)


#TODO change comment to english
# cherccher le meilleur nombre de voisin

L1=[]
L2=[]
for k in range(1,500,5):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(features,labels)
    predicted = knn_model.predict(ver)
    acc = accuracy_score(y_ver,predicted)
    L1.append(k)
    L2.append(acc)
max_value = max(L2)
max_index = L2.index(max_value)
best_n_neighbour = L1[max_index]

print(max_value)
print("best_neighbour is ",best_n_neighbour)
plt.plot(L1,L2)
plt.show()
