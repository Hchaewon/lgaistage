#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 패키지 로드
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os, random

from scipy import sparse
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from datetime import datetime
from datetime import timedelta


# In[2]:


# 경로 설정
data_path = '../data'
saved_path = './saved'
output_path = './submission'


# In[3]:


#bd = pd.read_csv('buy_data.csv')
#hd = pd.read_csv('history_data.csv')
mdp = pd.read_csv('meta_data_plus.csv')
md = pd.read_csv('meta_data.csv')
#prd = pd.read_csv('profile_data.csv')
#sd = pd.read_csv('search_data.csv')
#wd = pd.read_csv('watch_e_data.csv')
#ss = pd.read_csv('sample_submission.csv')


# In[4]:


# 데이터 불러오기 
history = pd.read_csv(os.path.join(data_path, 'history_data.csv'), encoding='utf-8')
watch = pd.read_csv(os.path.join(data_path, 'watch_e_data.csv'), encoding='utf-8')
buy = pd.read_csv(os.path.join(data_path, 'buy_data.csv'), encoding='utf-8') 
search = pd.read_csv(os.path.join(data_path, 'search_data.csv'), encoding='utf-8')
profile = pd.read_csv(os.path.join(data_path, 'profile_data.csv'), encoding='utf-8')
meta = pd.read_csv(os.path.join(data_path, 'meta_data.csv'), encoding='utf-8')
metaplus = pd.read_csv(os.path.join(data_path, 'meta_data_plus.csv'), encoding='utf-8')


# In[5]:


mt = meta.copy()
mtp = metaplus.copy()


# In[6]:


print(mt.shape)
print(mtp.shape)


# In[7]:


mt.head()


# In[8]:


mtp


# In[9]:


mt.info()


# In[10]:


mt.nunique()


# In[11]:


mtp.nunique()


# In[12]:


mt.country.unique()


# In[13]:


mt.genre_small.unique()


# In[18]:


mt_nocast = mt.drop(['cast_1','cast_2','cast_3','cast_4','cast_5','cast_6','cast_7','genre_small'],axis = 'columns')

mt_nocast


# ### 같은 album_id인데 빈값이 있으면 같은 항목으로 채워주기

# In[58]:


for i in mt_nocast['album_id'].value_counts().index : 
    
    if mt_nocast['album_id'].value_counts().loc[i] > 1 :
        if mt_nocast[mt_nocast['album_id']== i].country.isnull() is True : 
            print(mt_nocast[mt_nocast['album_id']== i])
    
print("done")
##같은 album_id인데 국가가 비어있는 항목은 없음


# ### meta데이터에서 Nan 값을 ‘기타’항목으로 변환

# In[64]:


mt_nocast[mt_nocast['country'].isnull() == True ]


# In[67]:


mt_nocast[mt_nocast['country'].isnull() == True ].replace(np.NaN, '기타')


# In[115]:


temp = mt_nocast
temp.isnull().sum()


# In[118]:


for i in temp[temp['country'].isnull()].index : 
    temp.loc[i,'country']= '기타'
    
    
temp.isnull().sum()


# In[119]:


temp


# In[ ]:





# In[120]:


temp.to_csv("pre_meta.csv", index = False)


# In[ ]:





# In[121]:


temp1 = pd.read_csv(os.path.join(data_path, 'pre_meta.csv'), encoding='utf-8')


# In[124]:


temp1.isnull().sum()


# In[ ]:





# In[ ]:




