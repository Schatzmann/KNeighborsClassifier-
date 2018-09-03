#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Autor:
#Annelyse Schatzmann         GRR20151731

import numpy as np
import scipy.interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import digits
from sklearn.datasets import load_svmlight_file
import os

import time
from sklearn.decomposition import TruncatedSVD


#------------- ÍRIS DATASET-------------#

iris = datasets.load_iris()

X = iris.data
# caracteristicas
y = iris.target
#classes

#------------- Cross -------------#

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

k_range = range(1, 31)
k_scores = []

for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X,y)
  scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
  k_scores.append(scores.mean())


plt.plot(k_range, k_scores, linestyle='-', color='blue', marker="*", label="Val. Cruzada")

#------------- Split -------------#
scores = []

for k in k_range :
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=0)
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	y_pred=knn.predict(X_test)
	scores.append(accuracy_score(y_test,y_pred))


plt.plot(k_range,scores, linestyle='-.', color='red', marker="o", label="40% Teste/60% Treinamento")

plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Valores de K')
plt.ylabel('Acuracia')
plt.title("Iris")
 
plt.show()


#------------- ACURÁCIA E KNN -------------#

def Cross(Xd, yd):

	X_train, X_test, y_train, y_test = train_test_split(Xd, yd, random_state=0)
	
	k_range = range(1, 31)
	k_scores = []
	
	for k in k_range:
	  knn = KNeighborsClassifier(n_neighbors=k)
	  knn.fit(X,y)
	  scores = cross_val_score(knn, Xd, yd, cv=5, scoring='accuracy')
	  k_scores.append(scores.mean())
	
	
	plt.plot(k_range, k_scores, linestyle='-', color='blue', marker="*", label="Val. Cruzada")



def Split(Xd, yd):
	scores = []

	for k in k_range :
		X_train, X_test, y_train, y_test = train_test_split(Xd, yd, test_size = 0.4, random_state=0)
		knn=KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train,y_train)
		y_pred=knn.predict(X_test)
		scores.append(accuracy_score(y_test,y_pred))
	
	
	plt.plot(k_range,scores, linestyle='-.', color='red', marker="o", label="40% Teste/60% Treinamento")

	plt.legend(loc='best')
	plt.grid(True)
	plt.xlabel('Valores de K')
	plt.ylabel('Acuracia')
		 


#------------- ACURÁCIA E TEMPO -------------#

def Cross_time(Xd, yd, tipo):


	X_train, X_test, y_train, y_test = train_test_split(Xd, yd, random_state=0)
	
	k_range = range(1, 31)
	k_scores = []
	time_c = []
	
	ini = time.time()

	for k in k_range:
	  knn = KNeighborsClassifier(n_neighbors=k, algorithm= tipo)
	  knn.fit(X,y)
	  scores = cross_val_score(knn, Xd, yd, cv=5, scoring='accuracy')
	  k_scores.append(scores.mean())
	  fim = time.time()
	  time_c.append(fim - ini)
		
	plt.plot(time_c, k_scores, linestyle='-', color='blue', marker="*", label="Val. Cruzada")



def Split_time(Xd, yd, tipo):
	scores = [] 
	time_s = []

	ini = time.time()

	for k in k_range :
		X_train, X_test, y_train, y_test = train_test_split(Xd, yd, test_size = 0.4, random_state=0)
		knn = KNeighborsClassifier(n_neighbors=k, algorithm= tipo)
		knn.fit(X_train,y_train)
		y_pred=knn.predict(X_test)
		scores.append(accuracy_score(y_test,y_pred))
		fim = time.time()
		time_s.append(fim - ini)

	plt.plot(time_s, scores, linestyle='-.', color='red', marker="o", label="40% Teste/60% Treinamento")

	plt.legend(loc='best')
	plt.grid(True)
	plt.xlabel('Tempo(s)')
	plt.ylabel('Acuracia')


#------------- DIGITS DATASET-------------#


def graph_digitos(Ximg, Yimg, tipo):

	arq = open ('features.txt', 'w')
	digitos = digits.load_images('./digits/data', arq, Ximg, Yimg)
	
	Xd, yd = load_svmlight_file('features.txt')
	# svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
	# svd.fit(Xd)
	# Xd = svd.transform(Xd)
	os.system("rm features.txt")

	if(tipo == ""):
		Cross(Xd, yd)
		Split(Xd, yd)
	else:
		Cross_time(Xd, yd, tipo)
		Split_time(Xd, yd, tipo)

	arq.close




graph_digitos(5,10,'')
plt.title("Digits(5x10)")
plt.show()
graph_digitos(20,10,'')
plt.title("Digits(20x10)")
plt.show()
graph_digitos(10,15,'')
plt.title("Digits(10x15)")
plt.show()
# graph_digitos(15,20,'')
# plt.title("Digits(15x20)")
# plt.show()
# graph_digitos(20,25,'')
# lt.title("Digits(20x25)")
# plt.show()


#------------- KD E BALL TREE -------------#
graph_digitos(10,15,'kd_tree')
plt.title("Kd Tree(10x15)")
plt.show()
graph_digitos(10,15,'ball_tree')
plt.title("Ball Tree(10x15)")
plt.show()
