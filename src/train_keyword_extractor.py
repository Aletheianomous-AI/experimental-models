#!/usr/bin/env python
# coding: utf-8

# In[1]:


import const
import pandas as pd
import torch

from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, TrainingArguments


# In[2]:


MODEL_OUTPUT_NAME = const.MODELS_FOLDER + "aletheianomous_ai-keyword_extractor-v0.2"


# In[3]:


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)


# In[4]:


training_df = pd.read_csv(const.DATASETS_FOLDER + "squad_ds_keyword_train.csv")
training_df = training_df[0:1000]
training_df = training_df.sample(frac=1)
zephyr = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-alpha", quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha", trust_remote_code=True)


# In[5]:


training_df.head()


# In[6]:


training_ds = Dataset.from_pandas(training_df)


# In[7]:


len(training_ds)


# In[8]:


len(training_df)


# In[9]:


zephyr.config.use_cache = False
zephyr.config.pretraining_tp = 1
zephyr.gradient_checkpointing_enable()


# In[10]:


zephyr


# In[11]:


tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
zephyr = prepare_model_for_kbit_training(zephyr)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)


# In[12]:


zephyr = get_peft_model(zephyr, peft_config)


# In[13]:


training_args = TrainingArguments(
    output_dir = const.MODELS_FOLDER + "/keyword-extractor",
    per_device_train_batch_size = 4,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm = 0.3,
    max_steps=-1,
    group_by_length=True,
    lr_scheduler_type="constant",
    num_train_epochs=3,
)

trainer = SFTTrainer(
    model=zephyr,
    train_dataset=training_ds, 
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    packing=False
)


# In[ ]:


trainer.train()


# In[ ]:


trainer.model.save_pretrained(MODEL_OUTPUT_NAME)
tokenizer.save_pretrained(MODEL_OUTPUT_NAME + "tokenizer")
zephyr.config.use_cache = True

