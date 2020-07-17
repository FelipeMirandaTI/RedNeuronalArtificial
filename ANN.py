# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:49:31 2020

@author: Felipe
"""
#Parte 1- Pre procesado de datos

#Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el data set
dataset = pd.read_csv(r'C:\Users\Felipe\Desktop\CursoDeepLearningPython\Churn_Modelling.csv')
X=dataset.iloc[:, [3,13]].values 
y=dataset.iloc[:, 13].values # valoriable dependiente o que quiero predecir
#las variables categoricas (string) no se pueden predecir, se deben convertir en variables dami (un array de 0 o 1)


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#El OneHotEncoder en las nuevas versiones está OBSOLETO
#onehotencoder = OneHotEncoder(categorical_features=[1])
#X = onehotencoder.fit_transform(X).toarray()

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("Churn_Modelling",        # Un nombre de la transformación
         OneHotEncoder(categories='auto'), # La clase a la que transformar
         [1]            # Las columnas a transformar.
         )
    ], remainder='passthrough'
)

X = transformer.fit_transform(X)
X = X[:, 1:] # Evitar la trampa de la multicolinealidad, es decir que no se creen muchos 
#arrays representado la misma variable


#Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Escalado de variables
# Si no se hace un escalado de variables puede confundir a la RN
# O sea hacer las varibles independiente y que ninguna variable se destaque sobre el resto
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#Parte 2 - Construir la RN

#importar keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la RNA
# Esta RNA va a clasificar
classifier= Sequential()

#Anadir las capas de entrada y primera capa oculta
#Dense conexion entre capas(unit (capas de la RN))
#Esta fase es expirimentar, en capa oculta debe ser el promedio entre la capa de entrada...
#y la  Capa de salida
#kernel_initializer funcion para inicializar los pesos de la RN
classifier.add(Dense(units=6,kernel_initializer="uniform",
                     activation="relu", input_dim = 11))
#input_dim cantidad de datos de entradas

#Añadir la segunda capa oculta
classifier.add(Dense(units=6,kernel_initializer="uniform",
                     activation="relu"))
#Anadir la capa de salida
classifier.add(Dense(units=1,kernel_initializer="uniform",
                     activation="sigmoid"))

#Compilar la RNA
# optimizador encuentra los pesos optimos para RNA
classifier.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])
#Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train,y_train, batch_size=10, epochs=100)
#batch_price => procesar 10 elementos y corregir 10 pesos

#Parte 3 - Evaluar el modelo y calcular predicciones finales
#Ajustar el clasificador en el conjunto de entrenamiento
#Crear el modelo de clasicacion aqui


#Prediccion de los resultados con el conjunto de testing
y_pred= classifier.predict(X_test)
y_pred= (y_pred>0.5)#si el valor es mayor a 0.5, quiere decir que el cliente abandonara el banco (probabilidad) 
#Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
#en la matriz de confusion quiere decir que la diagonal hacia abajo son los resultados correctos, y la diagonal 
#hacia abajo son los resultados negativos

#new_prediction = classifier.predict(sc_X.transform(np.array([[0,0,600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
#print(new_prediction)
#print(new_prediction > 0.5)
#[[......]] estos son los parametros para intentar predecir si ese cliente con esos parametros
#se puede ir del banco

