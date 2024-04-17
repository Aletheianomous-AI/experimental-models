#!/usr/bin/env python
# coding: utf-8

# # Test Keyword Extractor
# This function tests the keyword extractor model.

# In[1]:


from datasets import Dataset
from datetime import datetime as dt
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

import const
import evaluate
import numpy as np
import pandas as pd
import torch


# In[2]:


KW_MODEL_PATH = const.MODELS_FOLDER + "aletheianomous_ai-keyword_extractor-v0.1/"
TESTING_DS = const.DATASETS_FOLDER + "squad_ds_keyword_test.csv"


# In[3]:


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)


# In[4]:


kw_model = AutoModelForCausalLM.from_pretrained(KW_MODEL_PATH, quantization_config=bnb_config,
    device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(KW_MODEL_PATH + "tokenizer", trust_remote_code=True)


# In[5]:


tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True


# In[6]:


testing_ds = pd.read_csv(TESTING_DS)


# In[7]:


def apply_chat_template(user_prompt):
    prompt = ("<|system|>\nYou are a chatbot " +
              "that assists in providing " + 
              "information to the user. " + 
              "Please generate a keyword " + 
              "from the user's question.</s>\n<|user|>\n")
    prompt += user_prompt
    prompt += "</s>\n<|assistant|>\n"
    return prompt


# In[8]:


user_prompt = "Who created Five Night's at Freddys?"


# In[9]:


prompt = apply_chat_template(user_prompt)
kw_pipe = pipeline("text-generation", model=kw_model, tokenizer = tokenizer, torch_dtype=torch.bfloat16,
      device_map="auto")


# In[10]:


def generate_text(model_pipe, prompt):
    out = (model_pipe(prompt, do_sample=True, 
                  max_new_tokens=256, temperature=0.7,
                  top_k=50, top_p=0.95, num_return_sequences=1))
    #out = out[0]['generated_text']
    #out = out.split("<|assistant|>\n")
    #out = out[1]
    return out


# In[11]:


def prepare_prompts_for_val(val_feats):
    prompt_ls = []
    for prompt in val_feats:
        prompt_ls.append(apply_chat_template(prompt))
    #prompt_ls = np.array(prompt_ls)
    return prompt_ls


# In[12]:


def predictions_to_np(predictions):
    pred_ls = []
    percent = 0
    item_num = 1
    pred_len = len(predictions)
    for item in predictions:
        print("Converting predictions to sample...", percent, "% (", item_num, "/", pred_len, end="                      \r")
        pred_ls.append(item['generated_text'])
        item_num+=1
    pred_ls = np.array(pred_ls)
    return pred_ls


# In[13]:


def evaluate_kw(model_pipe, val_ds):
    """This function evaluates the keyword extractor model
       by the ROUGE metric.

        PARAMETERS
        model_pipe - The pipeline object that runs the keyword
            extractor model.
        val_ds - The dataset to evaluate the keyword extractor model.

    """

    print("EVALUATING MODEL...\n")
    ds_len = len(val_ds)
    text_labels = val_ds['text'].to_numpy()
    prompts = prepare_prompts_for_val(val_ds['question'].to_numpy())
    
    loss_f = evaluate.load('rouge')
    predictions = generate_text(model_pipe, prompts)
    predictions = predictions_to_np(predictions)
    
    
    losses = loss_f.compute(predictions=predictions, references=text_labels)
    return losses
        


# In[14]:


results = evaluate_kw(kw_pipe, testing_ds)


# In[ ]:


print("Validation loss metrics:")
print("Rouge-1 score: " + str(results['rouge1'] * 100) + "%")
print("Rouge-2 score: " + str(results['rouge2'] * 100) + "%")
print("Rouge-L score: " + str(results['rougeL'] * 100) + "%")
print("Rouge-L Sum score: " + str(results['rougeLsum'] * 100) + "%")

