#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1st task


# In[2]:


import pandas as pd


# In[3]:


import numpy as np
import sys
import struct
import re
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer


# In[4]:


filename = 'C:/agri_dataset_2020.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
words = text.split()
words = [word.lower() for word in words]


# In[5]:


words = re.split(r'dataFrame"":""', text)


# In[6]:


the_new_list = [x.split('""') for x in words]    
df= pd.DataFrame(the_new_list)


# In[7]:


df.drop(df.head(1).index, inplace=True)


# In[8]:


df.drop(df.columns.difference([0,14]), 1, inplace=True)


# In[9]:


soil_temp0 = []
soil_temp1 = []
soil_temp2 = []
room_temp = []
room_hum = []
rain_lvl= []
lumin = []
moisture = [] 
count=0
for i in df[0]:  
        soil_temp0.append(i[2:10])
        soil_temp1.append(i[10:18])
        soil_temp2.append(i[18:26])
        room_temp.append(i[26:34])
        room_hum.append(i[34:42])
        rain_lvl.append(i[42:46])
        lumin.append(i[46:50])
        moisture.append(i[50:54])   
time=df[14]


# In[10]:


df = pd.DataFrame(list(zip(soil_temp0,soil_temp1,soil_temp2,room_temp,room_hum,rain_lvl,lumin,moisture,time)),
                  columns =['soil_temp0', 'soil_temp1','soil_temp2', 'room_temp', 'room_hum', 
                            'rain_lvl','lumin', 'moisture','timestamps']) 


# In[11]:


import struct
soil_temp0 = df.soil_temp0.apply(lambda x: struct.unpack( '!f',bytearray.fromhex(x)))

soil_temp1 = df.soil_temp1.apply(lambda y: struct.unpack( '!f',bytes.fromhex(y)))
soil_temp2 = df.soil_temp2.apply(lambda x: struct.unpack( '!f',bytes.fromhex(x)))
room_temp = df.room_temp.apply(lambda x: struct.unpack( '!f',bytes.fromhex(x)))
room_hum = df.room_hum.apply(lambda x: struct.unpack( '!f',bytes.fromhex(x)))
rain_lvl = df.rain_lvl.apply(lambda x: int(x,16))
lumin = df.lumin.apply(lambda x: int(x,16))
moisture = df.moisture.apply(lambda x: int(x,16))


# In[12]:


df = pd.DataFrame(list(zip(soil_temp0,soil_temp1,soil_temp2,room_temp,room_hum,rain_lvl,lumin,moisture,time)),
                  columns =['soil_temp0', 'soil_temp1','soil_temp2', 'room_temp', 'room_hum', 
                            'rain_lvl','lumin', 'moisture','timestamps']) 


# In[13]:


df['soil_temp0'] = df['soil_temp0'].str[0]
df['soil_temp1'] = df['soil_temp1'].str[0]
df['soil_temp2'] = df['soil_temp2'].str[0]
df['room_temp'] = df['room_temp'].str[0]
df['room_hum'] = df['room_hum'].str[0]


# In[14]:


df


# In[15]:


df[['year', 'month', 'day']] = df['timestamps'].str.split('-', 2, expand=True)
df[['day', 'time']] = df['day'].str.split('T', 1, expand=True)
df=df.drop(columns=['timestamps'])


# In[16]:


df


# In[17]:


Feb = df.loc[(df['month'] == '02')]
March = df.loc[(df['month'] == '03')]


# In[ ]:





# In[18]:


Feb = Feb.iloc[:,:-1]
March = March.iloc[:,:-1]


# I have used imputation transformer to fill missing values with median values.     
# In my case, I have extracted without missing values, it is extraction error,that's why there is no change with imputation. 
# But I know, if there is missing values, which technique I will use.Now, there are less than 80% nan values, 
# so it will be better to impute with some mean or median 
# 

# In[19]:


imp = SimpleImputer(missing_values=-1, strategy='median')
a = imp.fit_transform(Feb.iloc[:,:-1])


# In[20]:


Feb.columns.values.tolist()


# In[21]:


from sklearn.model_selection import train_test_split

seed = 9001
np.random.seed(seed)
train_df , test_df = train_test_split(Feb ,test_size=0.2, random_state=seed)


# In[22]:


Y = train_df["soil_temp0"]
X = train_df[[ 'soil_temp1',
                 'soil_temp2',
                 'room_temp',
                 'room_hum',
                 'rain_lvl',
                 'lumin',
                 'moisture',
                 'year',
                 'month',
                 'day']]


# In[23]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=seed)


# In[24]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# In[25]:


Y_pred = regressor.predict(X_test)


# In[26]:


MF = pd.DataFrame({'Feb Actual': Y_test, 'Feb Predicted': Y_pred})
MF


# In[28]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))


# In[29]:


train_df , test_df = train_test_split(March ,test_size=0.2, random_state=seed)


# In[30]:


Y = train_df["soil_temp0"]
X = train_df[[ 'soil_temp1',
                 'soil_temp2',
                 'room_temp',
                 'room_hum',
                 'rain_lvl',
                 'lumin',
                 'moisture',
                 'year',
                 'month',
                 'day']]


# In[31]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=seed)


# In[32]:


regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# In[33]:


Y_pred = regressor.predict(X_test)


# In[34]:


MF = pd.DataFrame({'March Actual': Y_test, 'March Predicted': Y_pred})
MF


# In[35]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))


# In[ ]:





# ## 2nd task 
# Given the universities dataset (on Moodle), you are requested to perform the following tasks:
# 
# Apply an imputation algorithm of your choice to the missing data in the dataset (if any, and explain).
# Cluster Universities in this dataset into two groups, Private and Public, using a supervised learning approach. Please note that you don’t have to use the labels of 2nd column of the dataset (i.e., “yes” or “no”) to conclude to a solution.
# Compare the quality of the solution provided by the supervised learning algorithm to the labelled data. What is your conclusion?

# In[36]:


un  = pd.read_csv('C:/Universities.csv' )  


# In[37]:


un.head()


# In[38]:


import numpy as np
un1= un.iloc[:, 1:].replace(' ', np.nan, regex=True)


# In[39]:


un2 = un.iloc[:, :1]


# In[40]:


un = pd.concat([un1, un2], axis=1, sort=False)


# In[41]:


un.isnull().sum()


# In[42]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 0)
imp.fit(un)
imp.transform(un)


# In[43]:


un[:]= imp.transform(un)


# In[44]:


un


# In[45]:


un=un.drop(columns=['Unnamed: 0'])



# In[46]:


X = un.loc[:, un.columns != 'Private']
y = un['Private']


# In[47]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

un['Private'] = le.fit_transform(un['Private'])
#X['Unnamed: 0'] = le.fit_transform(X['Unnamed: 0'])
#s = le.transform(s)
y = le.transform(y)


# In[48]:


from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X = ms.fit_transform(X)


# In[49]:


X = pd.DataFrame(X)


# In[50]:


X.head()


# In[51]:


from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# In[53]:


x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size = 0.20)


# In[54]:


model = SVC()
model.fit(x_tr, y_tr)


# In[55]:


predict_private_public = model.predict(x_ts)
print('Accuracy: ', accuracy_score(y_ts, predict_private_public))


# In[56]:


y_pred = model.predict(x_ts)


# In[57]:


c = pd.DataFrame({'Actual': y_ts, 'Predicted': y_pred})
c


# In[ ]:




