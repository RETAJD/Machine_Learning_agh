#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)


# In[13]:


import pandas as pd
import numpy as np
X = mnist.data
y = pd.Series(mnist.target.astype(np.uint8))


# In[14]:


X


# In[15]:


print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[ ]:


#ZBIOR UCZACY I TESTOWY


# In[16]:


y1 = y.sort_values(ascending=True)


# In[17]:


y1


# In[18]:


X1 = X.reindex(y1.index)


# In[19]:


X1


# In[23]:


X_train, X_test = X1[:56000], X1[56000:]
y_train, y_test = y1[:56000], y1[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[30]:


y_train.unique()


# In[31]:


y_test.unique()


# In[32]:


#Zbiory nie są poprawne. Ze zbioru y_train algorytm nie nauczy sie rozpoznawac cyfr 8 i 9 i może mieć ciężko z cyfrą 7.
#Zbiory powinny być mniej więcej podobnie rozłożone, żeby algorytm mógł rozpoznać wszystkie cyfry.


# In[33]:


#dobrze rozłożone
X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[34]:


y_train.unique()


# In[35]:


y_test.unique()


# In[ ]:


# Uczenie jedna klasa


# In[36]:


y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)

print(y_train_0)
print(np.unique(y_train_0))
print(len(y_train_0))


# In[37]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)


# In[43]:


a = sgd_clf.predict(X_train)
b = sgd_clf.predict(X_test)


# In[45]:


# Policz dokładność ww. klasyfikatora na zbiorze uczącym oraz na zbiorze testującym.
import pickle
pickle.dump([sgd_clf.score(X_train, y_train_0), sgd_clf.score(X_test, y_test_0)], open("sgd_acc.pkl", "wb"))


# In[49]:


from sklearn.model_selection import cross_val_score

score = cross_val_score(sgd_clf, X_train, y_train_0,
                        cv=3, scoring="accuracy",
                        n_jobs=-1)
print(score)
#Policz 3-punktową walidację krzyżową dokładności (accuracy) modelu dla zbioru uczącego.
#Zapisz wynik (tablica) w pliku Pickle o nazwie sgd_cva.pkl.
pickle.dump(score, open("sgd_cva.pkl", "wb"))


# In[ ]:


#Uczenie, wiele klas


# In[50]:


sgd_m_clf = SGDClassifier(random_state=42,n_jobs=-1)
sgd_m_clf.fit(X_train, y_train)


# In[51]:


from sklearn.model_selection import cross_val_predict

#print(cross_val_score(sgd_m_clf, X_train, y_train,
#                      cv=3, scoring="accuracy", n_jobs=-1))
y_train_pred = cross_val_predict(sgd_m_clf, X_train,
                                 y_train, cv=3, n_jobs=-1)


# In[53]:


from sklearn.metrics import confusion_matrix

conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)

#Utwórz macierz błędów i zapisz ją w pliku Pickle o nazwie sgd_cmx.pkl.
pickle.dump(conf_mx, open("sgd_cmx.pkl", "wb"))

