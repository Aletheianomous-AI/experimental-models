#!/usr/bin/env python
# coding: utf-8

# # Generate Keywords Ground Truth
# This notebook generates the ground truth dataset for search keywords, given user input.

# In[ ]:


from multiprocessing import *
from transformers import pipeline
from datetime import datetime as dt

import const
import ctypes
import numpy
import pandas as pd
import torch
import traceback
import yake


# In[ ]:


KEYWORD_GEN_MODEL = "yake"


# In[ ]:


temp_ds = pd.read_csv(const.DATASETS_FOLDER + "squad-train-v2.0.csv")


# In[ ]:


k = Value(ctypes.py_object)
k.value = temp_ds


# In[ ]:


print(k.value)


# In[ ]:


def generate_keywords(ds_ns, start_row, end_row, batch_size = 64):
    #print("start: " + str(start_row) + ", end: " + str(end_row))
    temp_ds = ds_ns.df
    row_len = end_row - start_row
    if row_len <= batch_size:
        kw_extr = yake.KeywordExtractor(n=16)
        for row in range(start_row, end_row):
            keywords = kw_extr.extract_keywords(temp_ds.loc[row, "question"])
            if keywords == []:
                    print("No keywords extracted at row " + str(row) + " (question: " + temp_ds.loc[row, "question"] + ")")
            else:
                temp_ds.loc[row, "keyword"] = keywords[0][0]
            alt_keywords = []
            for item in keywords:
                alt_keywords.append(item[0])
            temp_ds.loc[row, "possible_keywords"] = str(alt_keywords)
        ds_ns.df = temp_ds
    else:
        row_range = end_row - start_row
        left_start = start_row
        left_end = int((row_range)/2) + left_start
        right_start = start_row + int((row_range)/2)
        right_end = end_row

        left_mgr = Manager()
        right_mgr = Manager()
        left_ns = left_mgr.Namespace()
        right_ns = right_mgr.Namespace()

        left_ns.df = ds_ns.df[0:int(row_range/2)].copy()
        right_ns.df = ds_ns.df[int(row_range/2):row_range].copy()
        p_left = Process(target=generate_keywords, args=(left_ns, left_start, left_end, batch_size))
        p_right = Process(target=generate_keywords, args=(right_ns, right_start, right_end, batch_size))
        p_left.start()
        p_right.start()
        p_left.join()
        p_right.join()
        ds_ns.df = pd.concat([left_ns.df, right_ns.df])


# In[ ]:


def generate_keywords_by_zephyr(model, ds):
    questions_processed = 0
    samples_len = len(ds)
    start_dt = dt.now()
    end_dt = None
    for row in range(len(ds)):
        content_msg = ("Hello Zephyr. I am creating a dataset that contains questions about topics. " + 
       "The questions may be asked by the user, thus it is a feature in the dataset I am creating. " +
       "Keywords are the dataset's label, which are keywords that can " +
       "be searched online to answer the user's questions. Please generate a kewyord that answers the following question in " +
       "quotation marks: \"")
        progress = int(questions_processed / samples_len) * 100
        progr_msg = ("Generating keywords... " + str(progress) + "% ("
            + "samples: " + str(questions_processed + 1) + "/" + str(samples_len))
        if end_dt is None:
            progr_msg += ", Elapsed time: 00:00.00)"
        else:
            elapsed_time = end_dt - start_dt
            avg_time = elapsed_time / questions_processed
            samples_remaining = samples_len - questions_processed
            time_remaining = avg_time * (samples_remaining)
            progr_msg += ", Elapsed time: " + str(elapsed_time) + ", "
            progr_msg += "Time remaining: " + str(time_remaining) + ")"
        print(progr_msg, end="                                                           \r")        
        question = ds.loc[row, "question"]
        content_msg += question + "\""
        msg = [{"role": "user", "content": content_msg}]
        prompt = zephyr.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        model_output = zephyr(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        model_output = model_output[0]
        model_output = model_output['generated_text']
        model_output = model_output.split("<|assistant|>\n")
        keyword = model_output[1]
        if keyword == None:
                print("No keywords extracted at row " + str(row) + " (question: " + ds.loc[row, "question"] + ")")
        else:
            ds.loc[row, "keyword"] = keyword
        end_dt = dt.now()
        questions_processed += 1


# In[ ]:


temp_ds.head()


# In[ ]:


if KEYWORD_GEN_MODEL == "yake":
    end = len(temp_ds)
    mgr = Manager()
    ns = mgr.Namespace()
    ns.df = temp_ds
    generate_keywords(temp_ds, 0, len(ns.df), batch_size=1024)
elif KEYWORD_GEN_MODEL == "zephyr":
    zephyr = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha",
                 torch_dtype = torch.bfloat16, device_map="auto")
    generate_keywords_by_zephyr(zephyr, temp_ds)
else:
    raise ValueError(KEYWORD_GEN_MODEL)


# In[ ]:


#zephyr = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha",
#                 torch_dtype = torch.bfloat16, device_map="auto")


# In[ ]:


#generate_keywords_by_zephyr(zephyr, temp_ds)


# In[ ]:


squad_ds.head()


# In[ ]:


squad_ds.tail()


# In[ ]:


squad_ds['keyword']


# In[ ]:


squad_ds['is_searchable'] = True


# In[ ]:


squad_ds.to_csv(const.DATASETS_FOLDER + "squad_ds_keyword_ground_truth.csv")

