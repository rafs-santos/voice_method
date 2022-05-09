import sys, os
sys.path.insert(0, './utils')

import matplotlib.pyplot as plt
import numpy as np

# Load Iris dataset
from sklearn import datasets
# Carregue cross-validation
from sklearn.model_selection import StratifiedKFold
# Carregue SVM
from sklearn.svm import SVC
# Carregue standardScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from plot_confusion import plot_confusion


if __name__ == '__main__':
# import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, 2:]  # we only take the first two features.
    y = iris.target
    target_names = iris.target_names
    target_names

    cross_val = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
    med = []
    teste_predct = []
    teste_class = []
    Mtx = []
    for train_indices, test_indices in cross_val.split(X,y):
        data_train = X[train_indices,:]
        data_teste = X[test_indices,:]

        label_train = y[train_indices]
        label_teste = y[test_indices]

        # Tira a média e dividi pelo desvio padrão para normalizar os dados
        scaler = StandardScaler()
        scaler.fit(data_train)

        data_train = scaler.transform(data_train)
        data_teste = scaler.transform(data_teste)

        # Cria-se o objeto SVM
        clf = SVC(gamma='auto', C=1000, kernel='rbf')
        clf.fit(data_train, label_train)

        y_class = clf.predict(data_teste)
        test_accuracy = np.mean(y_class.ravel() == label_teste.ravel()) * 100
        med.append(test_accuracy)
        Mtx.append(confusion_matrix(label_teste, y_class))

    a = sum(Mtx)
    print(sum(Mtx))
    fig, ax = plt.subplots(figsize=(12,8))
    plot_confusion(a, ax, ['setosa','versicolor', 'virginica'])
    plt.show()

