#!/usr/bin/env python
# coding: utf-8

# In[102]:


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


# In[116]:


df=pd.read_csv("diabetes.csv")


# In[117]:


df.head()


# In[118]:


data=df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]


# In[119]:


data=preprocessing.scale(data.values)


# In[120]:


data[0].shape


# In[121]:


labels=to_categorical(df["Outcome"].values)


# In[122]:


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.01)


# In[123]:


model=Sequential()
model.add(Dense(units=10,input_dim=8,activation="relu"))
model.add(Dense(units=100,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=100,activation="relu"))
model.add(Dense(units=2,activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[124]:


history=model.fit(x_train,y_train,epochs=100,shuffle=True)


# In[125]:


model.evaluate(x_train,y_train)


# In[126]:


model.predict_classes(x_test)


# In[127]:


res=[]
for point in y_test:
    if point[0]==1 and point[1]==0:
        res.append(0)
    elif point[0]==0 and point[1]==1:
        res.append(1)


# In[128]:


np.array(res)


# In[100]:


plt.plot(history.history["accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train','Test'], loc='upper left')
plt.show()


# In[101]:


plt.plot(history.history["loss"])
#plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train','Test'], loc='upper left')
plt.show()


# In[ ]:




