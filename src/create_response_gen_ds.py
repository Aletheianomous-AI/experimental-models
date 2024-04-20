#!/usr/bin/env python
# coding: utf-8

# # Create Response Generator Dataset
# This notebook creates the dataset for re-training the response generator model.
# 
# ## Pre-requisites
# A CSV file must be created that contains fields including input, search results, and output.

# In[1]:


from citation_fetcher import Citation_Fetcher as cf
from datetime import datetime as dt
from query_parser import QueryParser as QP
import const
import numpy as np
import pandas as pd


# In[2]:


RESPONSE_GEN_DS_PATH: str =  const.DATASETS_FOLDER + "response_gen_ground_truth.csv"
SQUAD_DS_PATH: str = const.DATASETS_FOLDER + "squad_ds_keyword_ground_truth.csv"
ULTRACHAT_LITE_PATH: str = const.DATASETS_FOLDER + "ultrachat_50k.csv"
REDDIT_DS_PATH: str = const.DATASETS_FOLDER + "reddit_ds.csv"


# In[3]:


squad_ds: pd.DataFrame = pd.read_csv(SQUAD_DS_PATH)
squad_len = len(squad_ds)


# In[4]:


ultrachat_ds: pd.DataFrame = pd.read_csv(ULTRACHAT_LITE_PATH)
ultrachat_len = len(squad_ds)


# In[5]:


reddit_ds: pd.DataFrame = pd.read_csv(REDDIT_DS_PATH)


# In[6]:


rg_ds = squad_ds['question']


# In[7]:


rg_ds = pd.DataFrame(data={'input': rg_ds, 'query': None, 'search_result': None, 'output': None})


# In[8]:


rg_ds.head()


# In[9]:


ultrachat_ds.head()


# In[10]:


ultrachat_ds = ultrachat_ds.rename(columns={'question': 'input'})


# In[11]:


rg_ds  = pd.concat([rg_ds, ultrachat_ds['input']], ignore_index=True)
rg_ds = rg_ds.sample(frac=1)
rg_ds = rg_ds.reset_index()
rg_ds = rg_ds[0:15000]


# In[14]:


i = 0
qp = QP(const.MODELS_FOLDER + "aletheianomous_ai-keyword_extractor-v0.3.1/")
start_time = dt.now()
elapsed_time = None
total_batch = len(rg_ds)
for j in range(len(rg_ds)):
    if elapsed_time is None:
        print("Populating Search Result ", (int((i/len(rg_ds))*100)),"% (Sample ", (i+1), "/", len(rg_ds) , ")", 
                end="                   \r")
    else:
        avg_time = elapsed_time / i
        time_remaining = avg_time * (total_batch-i)
        print("Populating Search Result ", (int((i/len(rg_ds))*100)),"% (Sample ", (i+1), "/", len(rg_ds) , "Elapsed time: ",
              elapsed_time, ", Time Remaining: ", time_remaining, ")",end="                                  \r")
    user_input = rg_ds.loc[j, "input"]
    query = qp.generate_query(str(user_input))
    json_out, result = cf.search_online(query)
    rg_ds.loc[i, "search_result"] = result
    i+=1
    end_time = dt.now()
    elapsed_time = end_time - start_time
    


# In[ ]:


reddit_ds.head()


# In[ ]:


reddit_ds = reddit_ds.rename(columns={"0": "input"})


# In[ ]:


rg_ds = pd.concat([rg_ds, reddit_ds['input']], ignore_index=True)


# In[ ]:


rg_ds.to_csv(RESPONSE_GEN_DS_PATH)

