#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[18]:


import os
os.chdir(r'C:\Users\Samuel Ephraim\Downloads\DATASET\Car Price Predictor')


# In[19]:


car_data = pd.read_csv('car data.csv')


# In[20]:


car_data.head()


# In[21]:


car_data.info()


# In[22]:


car_data.isnull().sum()


# In[23]:


car_data.describe()


# In[24]:


car_data.columns


# In[25]:


print(car_data['Fuel_Type'].value_counts())


# In[34]:


print(car_data['Seller_Type'].value_counts())
print(car_data['Transmission'].value_counts())


# In[35]:


print(car_data['Selling_Price'].value_counts())


# In[39]:


fuel_type = car_data['Fuel_Type']
seller_type = car_data['Seller_Type']
transmission_type = car_data['Transmission']  
selling_price = car_data['Selling_Price']


# In[40]:


from matplotlib import style


# In[41]:


style.use('ggplot')
fig = plt.figure(figsize=(15,5))
fig.suptitle('Visuallising Categorical Data')
plt.subplot(1,3,1)
plt.bar(fuel_type,selling_price, color='royalblue')
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price")
plt.subplot(1,3,2)
plt.bar(seller_type, selling_price, color='red')
plt.xlabel("Seller Type")
plt.subplot(1,3,3)
plt.bar(transmission_type, selling_price, color='purple')
plt.xlabel('Transmission type')
plt.show()


# In[42]:


fig, axes = plt.subplots(1,3,figsize=(15,5), sharey=True)
fig.suptitle('Visualizing categorical columns')
sns.barplot(x=fuel_type, y=selling_price, ax=axes[0])
sns.barplot(x=seller_type, y=selling_price, ax=axes[1])
sns.barplot(x=transmission_type, y=selling_price, ax=axes[2])


# In[43]:


petrol_data = car_data.groupby('Fuel_Type').get_group('Petrol')
petrol_data.describe()


# In[45]:


seller_data = car_data.groupby('Seller_Type').get_group('Dealer')
seller_data.describe()


# In[46]:


#manual encoding
car_data.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)


# In[47]:


#one hot encoding
car_data = pd.get_dummies(car_data, columns=['Seller_Type', 'Transmission'], drop_first=True)


# In[48]:


plt.figure(figsize=(10,7))
sns.heatmap(car_data.corr(), annot=True)
plt.title('Correlation between the columns')
plt.show()


# In[49]:


fig=plt.figure(figsize=(7,5))
plt.title('Correlation between present price and selling price')
sns.regplot(x='Present_Price', y='Selling_Price', data=car_data)


# In[50]:


X = car_data.drop(['Car_Name','Selling_Price'], axis=1)
y = car_data['Selling_Price']


# In[51]:


print("Shape of X is: ",X.shape)
print("Shape of y is: ", y.shape)


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

print("X_test shape:", X_test.shape)
print("X_train shape:", X_train.shape)
print("y_test shape: ", y_test.shape)
print("y_train shape:", y_train.shape)


# In[59]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LinearRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)


# In[57]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("MAE: ", (metrics.mean_absolute_error(pred, y_test)))
print("MSE: ", (metrics.mean_squared_error(pred, y_test)))
print("R2 score: ", (metrics.r2_score(pred, y_test)))


# In[60]:


sns.regplot(x=pred, y=y_test)
plt.xlabel("Predicted Price")
plt.ylabel('Actual Price')
plt.title("ACtual vs predicted price")
plt.show()


# In[ ]:




