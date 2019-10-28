#!/usr/bin/env python
# coding: utf-8

# # Classification using Tensorflow 

# ## Import required packages

# In[4]:


import pandas as pd
import tensorflow as tf


# # Loading Data set

# In[6]:


data = pd.read_csv("E:/diabetes.csv")


# In[7]:


data.head()


# In[8]:


data.isnull().any()


# In[9]:


import numpy as np
np.shape(data)


# In[10]:


data=data.dropna()


# In[11]:


data.isnull().any()


# In[12]:


np.shape(data)


# In[35]:


data.head()


# #### divide data in dependent and independent variables

# In[13]:


data.columns


# In[15]:


x=data.drop('Outcome',axis=1)


# In[17]:


y=data['Outcome']


# #### normalize data

# In[19]:


from sklearn.preprocessing import StandardScaler
x_s=StandardScaler()
y_s=StandardScaler()


# #### Split data into test and train

# In[20]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)


# In[21]:


xtrain=pd.DataFrame(data=x_s.fit_transform(xtrain),columns=xtrain.columns,index=xtrain.index)


# In[22]:


xtest=pd.DataFrame(data=x_s.fit_transform(xtest),columns=xtest.columns,index=xtest.index)


# In[23]:


data.columns


# #### creating feature columns

# In[24]:


Pregnancies = tf.feature_column.numeric_column("Pregnancies")


# In[25]:


Glucose = tf.feature_column.numeric_column("Glucose")


# In[26]:


BloodPressure = tf.feature_column.numeric_column("BloodPressure")


# In[27]:


SkinThickness = tf.feature_column.numeric_column("SkinThickness")


# In[28]:


Insulin = tf.feature_column.numeric_column("Insulin")


# In[29]:


BMI = tf.feature_column.numeric_column("BMI")


# In[30]:


DiabetesPedigreeFunction = tf.feature_column.numeric_column("DiabetesPedigreeFunction")


# In[32]:


Age = tf.feature_column.numeric_column("Age")


# In[34]:


input_list = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
       BMI, DiabetesPedigreeFunction, Age]


# In[38]:


xtrain = pd.DataFrame(data=x_s.fit_transform(xtrain),columns=xtrain.columns,index=xtrain.index)


# In[39]:


xtest = pd.DataFrame(data=x_s.fit_transform(xtest),columns=xtest.columns,index=xtest.index)


# In[ ]:


ytrain = pd.DataFrame(data=ytrain,index=ytrain.index)


# In[40]:


ytest = pd.DataFrame(data=ytest,index=ytest.index)


# In[41]:


input_layer =tf.estimator.inputs.pandas_input_fn(x=xtrain,y=ytrain,num_epochs=1000,batch_size=10,shuffle=True)


# In[49]:


model = tf.estimator.DNNClassifier(hidden_units=[10,20,20,20,10],feature_columns=input_list,n_classes=2)


# In[50]:


model.train(input_fn=input_layer,steps=1000)


# In[53]:


predict_layer = tf.estimator.inputs.pandas_input_fn(x=xtest,batch_size=10,shuffle=False,num_epochs=1)


# In[54]:


predict_values = model.predict(predict_layer)


# In[55]:


y_p = list(predict_values)


# In[56]:


y_p


# In[57]:


for i in y_p:
    print(i["class_ids"])


# In[ ]:




