#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


df = pd.read_csv("C:/Users/CYRIELLE/Documents/MLZoomcamp/Chap1/Homework/dataset.csv")


# In[6]:


df


# In[99]:


df.shape


# In[16]:


df.columns


# In[18]:


df.isnull().sum()


# In[34]:


df.Make.value_counts()


# In[49]:


df[df['Make'] == 'Audi'].nunique()


# In[50]:


z=df[df["Make"]=="Lotus"]
z


# In[101]:


z[['Engine HP', 'Engine Cylinders']]


# In[102]:


z[['Engine HP', 'Engine Cylinders']].drop_duplicates()


# In[107]:


z_drop=z[['Engine HP', 'Engine Cylinders']].drop_duplicates()
z_drop


# In[76]:


df["Engine Cylinders"].mean()


# In[90]:


df_fillna=df["Engine Cylinders"].fillna(5.628828677213059)
df_fillna


# In[91]:


df_fillna.mean()


# In[110]:


x=np.array([z_drop["Engine HP"],z_drop["Engine Cylinders"]])
x


# In[112]:


x_trans = x.transpose()
x_trans


# In[113]:


XTX= x_trans.dot(x)
XTX


# In[114]:


np.linalg.inv(XTX)


# In[115]:


y=np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])
y


# In[116]:


invXTX=np.linalg.inv(XTX)
invXTX


# In[117]:


res=invXTX.dot(x_trans)


# In[118]:


res


# In[120]:


w=y.dot(res)


# In[121]:


w

