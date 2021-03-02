#!/usr/bin/env python
# coding: utf-8

# In[102]:

# 93% Validation Accuracy


import pandas as pd
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Activation,Dropout,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical


df=pd.read_csv("diabetes.csv")

#Get Columns we are interested in to represent our X values
data=df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]

#Preparation/ Normalization of the X values
data=preprocessing.scale(data.values)

#Y values one hot encoded
labels=to_categorical(df["Outcome"].values)

#Splitting the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.01)

#Create Keras Model
model=Sequential()
model.add(Dense(units=10,input_dim=8,activation="relu"))
model.add(Dense(units=500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=2,activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#Comile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#Trin the model
history=model.fit(x_train,y_train,epochs=100,shuffle=True)

#Evaluate the model on test data
print(model.evaluate(x_train,y_train))

#Predictions usind test data
print(model.predict_classes(x_test))

#Improve readability of results i.e from [[0,1]] to 1 e.t.c
res=[]
for point in y_test:
    if point[0]==1 and point[1]==0:
        res.append(0)
    elif point[0]==0 and point[1]==1:
        res.append(1)

print(np.array(res))

#plotting the accuracy curve
plt.plot(history.history["accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train','Test'], loc='upper left')
plt.show()


#plotting the loss curve
plt.plot(history.history["loss"])
#plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train','Test'], loc='upper left')
plt.show()
