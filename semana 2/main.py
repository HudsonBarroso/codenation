#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[203]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# In[85]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[86]:


black_friday.head()


# In[116]:


black_friday.shape


# In[87]:


black_friday.describe


# In[88]:


black_friday.info()


# In[48]:


black_friday


# In[238]:


bool(len(black_friday.query("Product_Category_2 == 'NaN' & Product_Category_3 == 'NaN'")))


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[ ]:


def q1():
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[89]:


def q2():
    black_friday_M = (black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35')
    result = black_friday[black_friday_M].shape[0]
    return result


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[ ]:


def q3():
    black_friday['User_ID'].isnull().sum()
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[ ]:


def q4():
    return black_friday.dtypes.unique().shape[0]


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[ ]:


def q5():
    return black_friday[black_friday.isna().any(axis=1)].shape[0] / black_friday.shape[0]


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[ ]:


def q6():
    return black_friday.shape[0] - black_friday.dropna().shape[0]


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[ ]:


def q7():
    return black_friday['Product_Category_3'].value_counts().idxmax()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[ ]:


def normal(x):
    return (x - x.min()) / (x.max() - x.min())
    
def q8():
    return float(normal(black_friday.Purchase).mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[ ]:


def q9():
    standardized = (black_friday['Purchase'] - black_friday['Purchase'].mean()) / black_friday['Purchase'].std()
    standardized.between(-1, 1)
    return int(standardized.between(-1, 1).sum())


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[ ]:


def q10():
    return bool(len(black_friday.query("Product_Category_2 == 'NaN' & Product_Category_3 == 'NaN'")))

