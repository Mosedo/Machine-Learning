#!/usr/bin/env python
# coding: utf-8

# In[45]:


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


# In[46]:


dataframe = pd.read_csv("pokemon_data.csv")


# In[47]:


dataframe.head(3)


# In[48]:


dataframe.isnull().values.any()


# In[49]:


#df=pd.read_excel("https://drive.google.com/file/d/1CI7ilGn02qi2mbIlEcKqMasUfXlz-gN0/view?usp=sharing")


# In[50]:


dataframe.loc[dataframe['Attack'] <= 50] #get rows with conditions


# In[51]:


#dataframe.head().values #Converting Dataframe to numpy array


# In[52]:


#dataframe["Total"]=20 #adding a column to the dataframe


# In[53]:


#dataframe.drop(["Name"],axis=1,inplace=True) #Dropping a dataframe column


# In[54]:


dataframe.isna().sum() #Number of missing values per column


# In[55]:


dataframe.head()


# In[56]:


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


# In[57]:


#dataframe["Legendary"].unique()
dataframe.head()


# In[58]:


data=dataframe[["Type 1","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed","Generation"]]


# In[59]:


data=preprocessing.scale(data.values)


# In[60]:


labels=to_categorical(dataframe["Legendary"].values)


# In[61]:


x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.3)


# In[62]:


model=Sequential()
model.add(Dense(units=10,input_dim=8,activation="relu"))
model.add(Dense(units=2,activation="softmax"))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[63]:


history=model.fit(x_train,y_train,epochs=100,validation_data=(x_test, y_test),shuffle=True,verbose=2)


# In[64]:


model.evaluate(x_test,y_test)


# In[65]:


prediction=model.predict_classes(x_test)


# In[67]:


prediction


# In[68]:


res=[]
for point in y_test:
    if point[0]==1 and point[1]==0:
        res.append(0)
    elif point[0]==0 and point[1]==1:
        res.append(1)


# In[69]:


true_labels=np.array(res)


# In[71]:


true_labels


# In[72]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=2)
from sklearn.metrics import confusion_matrix


# In[73]:


confusion_matrix(prediction, true_labels)


# In[82]:


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train','Test'], loc='lower right')
plt.show()


# In[83]:


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train','Test'], loc='upper right')
plt.show()


# In[85]:


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[90]:


mat=confusion_matrix(true_labels,prediction)
plot_confusion_matrix(conf_mat=mat,figsize=(7,7))


# In[ ]:




