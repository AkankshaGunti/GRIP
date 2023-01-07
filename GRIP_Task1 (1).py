#!/usr/bin/env python
# coding: utf-8
BY GUNTI AKANKSHA 
The Sparks Foundation
DataScience & Business Analytics Intern
GRIPJANUARY23
Task-1: Prediction using Supervised MachineLearning
For this we have to apply Linear Regression for predicting student's percentage based on the number.of study hours
# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[4]:


df.head()


# In[5]:


df.isnull()


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# In[8]:


columns = list(df.columns)


# In[9]:


from sklearn import preprocessing


# In[12]:


X = df["Hours"].values.reshape(-1,1)
Y = df["Scores"].values.reshape(-1,1)


# In[26]:


plt.scatter(X,Y)
plt.title("Hours Vs Scores")
plt.grid()
plt.show()


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)


# In[29]:


from sklearn.linear_model import LinearRegression


# In[30]:


lr = LinearRegression()
lr


# In[31]:


lr.fit(x_train,y_train)


# In[32]:


line = lr.coef_ * X + lr.intercept_


# In[33]:


plt.scatter(x_train , y_train , color = "#329ba8")


# In[35]:


plt.scatter(x_train , y_train , color = "#329ba8")
plt.plot(X , line , color = "r")
plt.show()


# In[36]:


Y_pred = lr.predict(x_test)
Y_pred


# In[37]:


plt.scatter(x_test,y_test , color = "#75a6eb")
plt.plot(x_test,Y_pred , color = "black")
plt.show()


# In[38]:


df_predict = pd.DataFrame({"Hours": x_test.reshape(1,-1)[0] , "Actual Score" : y_test.reshape(1,-1)[0] , "Predicted Score" : Y_pred.reshape(1,-1)[0]})
df_predict


# In[39]:


df_sorted = df_predict.sort_values(by = "Hours")
df_sorted


# In[40]:


from sklearn.metrics import r2_score
from sklearn import metrics

mean_absolute_error=metrics.mean_absolute_error(y_test,Y_pred)
print('Mean absolute error:',mean_absolute_error)

corr=r2_score(y_train,lr.predict(x_train))
print('correlation:',corr)

acc=r2_score(y_test,Y_pred)
print('Accuracy:',acc)


# In[41]:


hrs = 9.25
pred = lr.predict([[9.25]])
print("The predicted score if a student studies for 9.25 hrs/ day is",pred[0])


# In[ ]:




