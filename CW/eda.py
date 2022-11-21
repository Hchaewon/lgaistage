#!/usr/bin/env python
# coding: utf-8

# In[158]:


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


# In[5]:


hd = history.copy()
wd = watch.copy()
bd = buy.copy()
sd = search.copy()
pd = profile.copy()


# In[6]:


print('hd 데이터(중복 제거 전) : ', hd.shape)
print('wd 데이터(중복 제거 전) : ', wd.shape)


# #### 1. 중복 데이터 제거

# In[7]:


# 중복행 확인
hd[hd.duplicated()] # 시청시작 데이터


# In[8]:


wd[wd.duplicated()] # 시청종료 데이터


# In[9]:


# 중복행 제거
hd = hd[~hd.duplicated()]
print('hd 데이터(중복 제거 후) : ', hd.shape)

wd = wd[~wd.duplicated()]
print('wd 데이터(중복 제거 후) : ', wd.shape)


# #### 2. 이상치 제거

# #### 2-1. History_data 내 log_time 이상치 제거

# In[10]:


hd.head()


# In[11]:


# log_time 날짜/시간 분리
hd["log_time"] = hd["log_time"].astype(str)
hd["date"] = hd["log_time"].str.slice(0, 8)
hd["time"] = hd["log_time"].str.slice(8, 14)


# In[12]:


hd.head()


# In[13]:


hd.info()


# In[14]:


# 월 조건 (3 ~ 7월 제외한 월이 있는지)
con1 = (hd['date'].str[4:6] != '03')
con2 = (hd['date'].str[4:6] != '04')
con3 = (hd['date'].str[4:6] != '05')
con4 = (hd['date'].str[4:6] != '06')
con5 = (hd['date'].str[4:6] != '07')

# 일 조건 (0일 / 00일 / 32일 이상 있는지)
con8 = (hd['date'].str[6:8] == '0')
con9 = (hd['date'].str[6:8] == '00')
con10 = (hd['date'].str[6:8].astype(int) >= 32)


# In[15]:


print('조건에 맞지 않는 연도 수:', len(hd.loc[hd['date'].str[:4] != '2022']))
print('조건에 맞지 않는 월 수:', len(hd.loc[con1 & con2 & con3 & con4 & con5]))
print('조건에 맞지 않는 일 수:', len(hd.loc[con8 | con9 | con10]))


# In[16]:


print('조건에 맞지 않는 시 수:', len(hd.loc[hd['time'].str[:2] >= '24']))
print('조건에 맞지 않는 분 수:', len(hd.loc[hd['time'].str[2:4] >= '60']))
print('조건에 맞지 않는 초 수:', len(hd.loc[hd['time'].str[4:6] >= '60']))


# In[17]:


# 초에만 이상 있음
# 이상 있는 데이터 73,581개
hd.loc[hd['time'].str[4:6] >= '60']


# In[18]:


# 이상 데이터 제거
hd = hd.drop(hd.loc[hd['time'].str[4:6] >= '60'].index)
print('이상 데이터 제거 후:', len(hd))


# In[19]:


print('조건에 맞지 않는 초 수:', len(hd.loc[hd['time'].str[4:6] >= '60']))


# #### 3. 결측치 제거 (payment Nan 값을 0으로)

# In[20]:


# 결측치 확인
# wd, bd, pd에는 결측치x / pd keyword에 결측치 존재
hd.isnull().sum()


# In[21]:


hd.replace(np.nan,0,inplace = True)


# In[22]:


hd.isnull().sum()


# In[23]:


hd.head()


# In[24]:


hd.to_csv("pre_hd.csv")


# In[ ]:





# In[ ]:





# #### 4. 파생변수 (선호도) 

# In[57]:


wd.tail()


# In[36]:


#컬럼추가

wd_prefer = wd.copy()
wd_prefer['prefer'] = np.nan
wd_prefer.head()


# In[98]:


#조건걸어서
#새컬럼에 값추가 1 ~ 4

for i in wd_prefer.index : 
    
    try : 
        #wd_prefer_check = wd_prefer.iloc[i]
        if wd_prefer.loc[i].watch_time < wd_prefer.loc[i].total_time * 0.25 :
            wd_prefer.loc[i,'prefer'] = 1
        elif wd_prefer.loc[i].watch_time < wd_prefer.loc[i].total_time * 0.5 :
            wd_prefer.loc[i,'prefer'] = 2
        elif wd_prefer.loc[i].watch_time < wd_prefer.loc[i].total_time * 0.75 :
            wd_prefer.loc[i,'prefer'] = 3
        else : 
            wd_prefer.loc[i,'prefer'] = 4

        
    except : 
        pass


# In[99]:


wd_prefer.head()


# In[110]:


### prefer에 nan있는지 확인

for i in wd_prefer.index 
    if wd_prefer.loc[i].prefer is not None == False : 
        print(i)


# In[ ]:





# ### 5. History data 유실부분 Watch data에서 append

# #### 5-1 .history, watch data 확인

# In[26]:


# 결측치 처리방법이 완전히 정해진게 아니므로 여기서 가격이 0 처리된 history data를 이용해서 작업함
# 결측치 처리방법이 정해지면 코드 뽑아서 수정


# In[111]:


from datetime import datetime


# In[112]:


hd.head()


# In[113]:


wd.head()


# In[114]:


print(hd.shape)
print(wd.shape)


# ### 5-2. history 유실데이터 확인

# In[115]:


hd_pid = hd['profile_id'].value_counts(dropna=False).sort_index()
wd_pid = wd['profile_id'].value_counts(dropna=False).sort_index()

print("\n\nhd_pid\n", hd_pid)
print("\n\nwd_pid\n", wd_pid)


# #### 문의에 나온 24번 profile 의 문제상황 확인

# In[116]:


for i in range( hd.shape[0]) : 
    if hd.iloc[i].profile_id == 24 :
        print("\n--------\n")
        print("i : ", i )
        print(hd.iloc[i])
    else :
        pass
    
    
#loc : 전체 데이터 프레임에서 인덱스 이름이 0인 행만 추출
#iloc : 전체 데이터 프레임에서 0번쨔 행에 있는 값들만 추출


# In[117]:


for i in range( wd.shape[0]) : 
    if wd.iloc[i].profile_id == 24 :
        print("\n--------\n")
        print("i : ", i )
        print(wd.iloc[i])
    else :
        pass
    


# In[ ]:





# ####

# In[243]:


#datetime 변환 함수(초까지)
datetime_format = "%Y%m%d%H%M%S"

def logtodate(log) :
    datetime_string = str(log)
    datetime_result = datetime.strptime(datetime_string, datetime_format)
    #print(datetime_result)
    return datetime_result

def logtodateint(log) :
    datetime_int = str(log)
    datetime_result = datetime.strptime(datetime_int, datetime_format)
    #print(datetime_result)
    return datetime_result


# In[ ]:





# In[245]:


##logtime 비교해서 유실데이터 찾는 코드 

losscnt = 0
is_album_hd = True


for i in wd_pid.index : 
    tmp_wd = wd[wd['profile_id'] == i]
    tmp_hd = hd[hd['profile_id'] == i]
    
    
    #album으로한번 더 포문 돌려서 나누기 x2
    for j in tmp_wd.album_id : 
        try : 
            tmp_wd_album = tmp_wd[tmp_wd['album_id'] == j]
            tmp_hd_album = tmp_hd[tmp_hd['album_id'] == j]
        
        except :
            print("album_id : ", j ," 는 HD에 존재하지 않음")
            is_album_hd = False
            pass
        
        if is_album_hd == False :
            
            is_album_hd = True
            pass
        
        else : 
            
            for k in tmp_hd_album.index:
                log = logtodate(tmp_hd_album.log_time.loc[k])
                #tmp_hd.log_time.iloc[j] = round(logtodate(log),-1)
                #tmp_hd_album.log_time.loc[k] = logtodate(log) #datetime 형태
            
            for k in tmp_wd_album.index : 
                log = logtodateint(tmp_wd_album.log_time.loc[k])
                #log = logtodate(tmp_wd_album.log_time.loc[k])
                #tmp_wd.log_time.iloc[j] = round(logtodate(log),-1)
                #tmp_wd_album.log_time.loc[k] = logtodate(log) #datetime 형태
                #log = tmp_wd_album.log_time.loc[k]
                
                watchtime = tmp_wd.watch_time.loc[k]  #int형               
                #comparetime = log - timedelta(seconds=int(watchtime))
                ##4중포문 이대로도 괜찮은가
                
                for h in tmp_hd_album.index : 
                    cnt = 0 
                    
                    log_hd = logtodate(tmp_hd_album.log_time.loc[h])
                    
                    comparetime = log - log_hd #초 단위의 datetime
                    comparetime_round = round(comparetime.seconds,-1) #1의 자리수에서 반올림
                    if comparetime == watchtime or comparetime_round == watchtime  : 
                        cnt += 1
                        
                if cnt < 1 : 
                    print("---------")
                    print("k : ", k )
                    print(tmp_wd_album.loc[k])
                    losscnt += 1
        
        
    
    

            
        
print("loss cnt : ", losscnt)
        
        


# In[ ]:


#### 정제후 데이터로한거 유실데이터 20454개 

cnt = 0

for i in wd_pid.index : 
    tmp_wd = wd[wd['profile_id'] == i]
    tmp_hd = hd[hd['profile_id'] == i]
    
    
    #album으로한번 더 포문 돌려서 나누기 x2
    for j in tmp_wd.album_id : 
        
        tmp_wd_album = tmp_wd[tmp_wd['album_id'] == j]
        tmp_hd_album = tmp_hd[tmp_hd['album_id'] == j]
        
        if tmp_hd_album.index.shape[0] == 0 :
            cnt += 1 
            print(cnt ," : album_id ", j ," 는 HD profile_id ", i ,"에 존재하지 않음")
            is_album_hd = False
            


# In[291]:


###정제 전 데이터로 한거

cnt = 0

for i in wd_pid.index : 
    tmp_wd = watch[watch['profile_id'] == i]
    tmp_hd = history[history['profile_id'] == i]
    
    
    #album으로한번 더 포문 돌려서 나누기 x2
    for j in tmp_wd.album_id : 
        
        tmp_wd_album = tmp_wd[tmp_wd['album_id'] == j]
        tmp_hd_album = tmp_hd[tmp_hd['album_id'] == j]
        
        if tmp_hd_album.index.shape[0] == 0 :
            cnt += 1 
            print(cnt ," : album_id ", j ," 는 HD profile_id ", i ,"에 존재하지 않음")
            is_album_hd = False
            


# In[280]:


hd[hd['profile_id']==33032]


# In[281]:


wd[wd['profile_id']==33032]


# In[290]:


history[history['profile_id'] == 33016].shape


# In[289]:


watch[watch['profile_id'] == 33016].shape


# In[ ]:





# In[293]:


ss = history[history['profile_id'] == 33032]
ss = ss[ss['album_id'] == 373] 


# In[296]:


dd = watch[watch['profile_id'] == 33032]
dd = dd[dd['album_id'] == 373]

dd


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




