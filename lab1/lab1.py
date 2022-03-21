#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib 
import urllib.request
import tarfile
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


os.mkdir("/Data")
os.chdir("/Data")


# In[3]:


#pobranie danych z linku
urllib.request.urlretrieve("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz", "housing.csv.gz")


# In[4]:


#wypakowanie z tar
tar = tarfile.open("housing.csv.gz")
tar.extractall()
tar.close()


# In[5]:


#kompresja do GZIP
with open('housing.csv', 'rb') as f_in, gzip.open('housing.csv.gz', 'wb') as f_out:
    f_out.writelines(f_in)


# In[6]:


#robienie dataframe
df = pd.read_csv('housing.csv.gz')
#zmienna df to Dataframe - obiekt gdzie trzymamy dane 

#df.head()

#df.loc[:,["ocean_proximity"]].value_counts()

#df.loc[:,["ocean_proximity"]].describe()


# In[7]:


df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')


# In[8]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.1, figsize=(7,4))
plt.savefig('obraz2.png')


# In[9]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.4, figsize=(7,3), colorbar=True,
        s=df["population"]/100, label="population", 
        c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')


# In[10]:


# ANALIZA
df.corr()["median_house_value"].sort_values(ascending=False)


# In[11]:


data1 = df.corr()["median_house_value"].sort_values(ascending=False)
data1 = data1.reset_index().rename(columns={"index":"atrybuty", "median_house_value":"wspolczynnik_korelacji"})

data1.to_csv('korelacja.csv')


# In[12]:


sns.pairplot(df)


# In[13]:


#Przygotowanie do uczenia
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
len(train_set),len(test_set)

print(train_set)
print(test_set)



# In[14]:


#macierz korelacji dla train_set
train_set.corr()["median_house_value"].sort_values(ascending=False)


# In[15]:


#macierz korelacji dla test_set
test_set.corr()["median_house_value"].sort_values(ascending=False)


# In[16]:


#to picle file: train_set
train_set.to_pickle("train_set.pkl")  


# In[17]:


#to picle file: test_set
test_set.to_pickle("test_set.pkl")  


# In[ ]:




