#!/usr/bin/env python
# coding: utf-8

# ### Importing Needed packages
# 

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing Os and The Location of the dataset
# 

# In[2]:


import os
os.chdir(r'C:\Users\Samuel Ephraim\Desktop\Ai Labs\Machine Learning With Python\Regression(Week 2)')


# ## Reading the data in
# 

# In[3]:


df = pd.read_csv('FuelConsumptionCo2.csv')


# In[4]:


df.head()


# ### Data Exploration
# 
# Let's first have a descriptive exploration on our data.
# 

# In[5]:


# summarize the data
df.describe()


#  Selecting some features to explore more.
# 

# In[6]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


# Plotting each of these features:
# 

# In[7]:


viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


# Plotting each of these features against the CO2Emissions, to see how linear their relationship is:
# 

# In[8]:


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()


# In[9]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# ### Creating a training and testing dataset
# 
# Train/Test Split involves dividing the dataset into mutually exclusive training and testing sets. The training set is then used to train, and the testing set is used to test.
# Because the testing dataset is not part of the dataset used to train the model, this will provide a more accurate evaluation of out-of-sample accuracy. As a result, we have a better understanding of how well our model generalizes to new data.
# 
# This means we know the outcome of each data point in the testing dataset, which makes it ideal for testing! Because this data was not used to train the model, the model has no idea how these data points will turn out. In essence, this is true out-of-sample testing.
# 
# Divide our dataset into train and test sets. 80% of the total dataset will be used for training, while 20% will be used for testing. Using the **np.random.rand()** function, we create a mask to select random rows:

# In[10]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# #### Train data distribution
# 

# In[11]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# #### Modeling
# 
# Using sklearn package to model data.
# 

# In[12]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# As previously stated, the parameters of the fit line in simple linear regression are **Coefficient** and **Intercept**.
# Sklearn can estimate the intercept and slope of the line directly from our data because it is a simple linear regression with only two parameters.
# It is important to note that all of the data must be available in order to traverse and calculate the parameters.

# #### Plot outputs
# 

# We can plot the fit line over the data:
# 

# In[13]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# #### Evaluation
# To calculate the accuracy of a regression model, we compare the actual and predicted values. Evaluation metrics play an important role in the development of a model because they reveal areas that need to be improved.
# 
# There are various model evaluation metrics; we'll use MSE here to calculate our model's accuracy based on the test set:
# 
# * Mean Absolute Error: This is the average of the errors' absolute values. Because it is simply average error, this is the simplest of the metrics to grasp.
# 
# * Mean Squared Error (MSE): The mean of the squared error is the mean squared error (MSE). It is more popular than Mean Absolute Error because it focuses on large errors.This is because the squared term increases larger errors exponentially more than smaller ones.
# 
# * Root Mean Squared Error (RMSE).
# 
# * R-squared is a popular metric for measuring the performance of your regression model, not an error. It represents the distance between the data points and the fitted regression line. The better the model fits your data, the higher the R-squared value. The highest possible score is 1.0, but it can also be negative (because the model can be arbitrarily worse).

# In[14]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )


# ### MISC

# In[15]:


train_x = train[["FUELCONSUMPTION_COMB"]]

test_x = test[["FUELCONSUMPTION_COMB"]]

regr = linear_model.LinearRegression()

regr.fit(train_x, train_y)

predictions = regr.predict(test_x)

print("Mean Absolute Error: %.2f" % np.mean(np.absolute(predictions - test_y)))


# In[ ]:




