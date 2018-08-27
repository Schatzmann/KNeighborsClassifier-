#coding: utf-8

#Autor:
#Annelyse Schatzmann           GRR20151731

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import digits
from sklearn.datasets import load_svmlight_file


#------------- ÍRIS DATASET-------------#

iris = datasets.load_iris()

X = iris.data
# caracteristicas
y = iris.target
#classes

#------------- Melhor k para 5 folds-------------#

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

k_range = range(1, 31)
k_scores = []

for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X,y)
  scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
  k_scores.append(scores.mean())


plt.plot(k_range, k_scores)
plt.xlabel('Valores de K')
plt.title('Validacao Cruzada- Iris')
plt.ylabel('Acuracia')

plt.show()

#------------- Melhor k para 30 execuções-------------#
scores = []

for k in k_range :
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=0)
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	y_pred=knn.predict(X_test)
	scores.append(accuracy_score(y_test,y_pred))


plt.plot(k_range,scores)
plt.xlabel('Valores de K')
plt.title('40% Teste e 60% Treinamento- Iris')
plt.ylabel('Acuracia')

plt.show()

#------------- DIGITS DATASET-------------#

arq = open ('features.txt', 'w')
digitos = digits.load_images('./digits/data', arq)


Xd, yd = load_svmlight_file('./features.txt')


#------------- Melhor k para 5 folds-------------#

X_train, X_test, y_train, y_test = train_test_split(Xd, yd, random_state=0)

k_range = range(1, 31)
k_scores = []

for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X,y)
  scores = cross_val_score(knn, Xd, yd, cv=5, scoring='accuracy')
  k_scores.append(scores.mean())


plt.plot(k_range, k_scores)
plt.xlabel('Valores de K')
plt.title('Validacao Cruzada- Digitos')
plt.ylabel('Acuracia')

plt.show()

#------------- Melhor k para 30 execuções-------------#
scores = []

for k in k_range :
	X_train, X_test, y_train, y_test = train_test_split(Xd, yd, test_size = 0.4, random_state=0)
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	y_pred=knn.predict(X_test)
	scores.append(accuracy_score(y_test,y_pred))


plt.plot(k_range,scores)
plt.xlabel('Valores de K')
plt.title('40% Teste e 60% Treinamento- Digitos')
plt.ylabel('Acuracia')

plt.show()
