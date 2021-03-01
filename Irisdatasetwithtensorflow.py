#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import numpy as np

iris = load_iris()
X = preprocessing.scale(iris['data'])
Y = to_categorical(iris['target'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = Sequential()
model.add(Dense(6, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, batch_size=10)

print(model.evaluate(X_test,Y_test))


print(model.predict_classes(X_test))

res=[]
for point in Y_test:
    if point[0]==1 and point[1]==0 and point[2]==0:
        res.append(0)
    elif point[0]==0 and point[1]==1 and point[2]==0:
        res.append(1)
    elif point[0]==0 and point[1]==0 and point[2]==1:
        res.append(2)
print(np.array([res]))


