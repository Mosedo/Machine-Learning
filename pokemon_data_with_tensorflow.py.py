#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import Adam,SGD
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import numpy as np


dataframe = pd.read_csv("pokemon_data.csv")


dataframe["Type 1"]=dataframe["Type 1"].map({
    "Grass": 0,
    "Fire": 1,
    "Water": 2,
    "Bug": 3,
    "Normal": 4,
    "Poison": 5,
    "Electric": 6,
    "Ground": 7,
    "Fairy": 8,
    "Fighting": 9,
    "Psychic": 10,
    "Rock": 11,
    "Ghost": 12,
    "Ice": 13,
    "Dragon": 14,
    "Dark": 15,
    "Steel": 16,
    "Flying": 17
})

dataframe["Legendary"]=dataframe["Legendary"].map({
    False: 0,
    True: 1
})

data=dataframe[["Type 1","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed","Generation"]]
data=preprocessing.scale(data.values)

labels=to_categorical(dataframe["Legendary"].values)

x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.3)

model=Sequential()
model.add(Dense(units=10,input_dim=8,activation="relu"))
model.add(Dense(units=2,activation="softmax"))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

history=model.fit(x_train,y_train,epochs=100,validation_data=(x_test, y_test),shuffle=True,verbose=2)

model.evaluate(x_test,y_test)

prediction=model.predict_classes(x_test)

res=[]
for point in y_test:
    if point[0]==1 and point[1]==0:
        res.append(0)
    elif point[0]==0 and point[1]==1:
        res.append(1)

true_labels=np.array(res)

true_labels


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=2)
from sklearn.metrics import confusion_matrix

confusion_matrix(prediction, true_labels)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train','Test'], loc='lower right')
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train','Test'], loc='upper right')
plt.show()


#Plot Confusion Matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(true_labels,prediction)
plot_confusion_matrix(conf_mat=mat,figsize=(7,7),class_names=["Legendary","Not Legendary"],cmap=plt.cm.Blues,hide_spines=True,hide_ticks=False)
