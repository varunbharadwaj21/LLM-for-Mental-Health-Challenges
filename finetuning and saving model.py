#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import json


# In[3]:


with open('/content/mentalhealth.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data['intents'])
df


# In[4]:


dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)

df = pd.DataFrame.from_dict(dic)
df


# # Leveraging LLM

# In[16]:


pip install datasets


# In[17]:


# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import json


# In[18]:


dataset = load_dataset("json", data_files="mentalhealth.json")


# In[19]:


# Load a pre-trained tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# In[20]:


# Setting the padding token to the EOS token for GPT-2
tokenizer.pad_token = tokenizer.eos_token


# In[21]:


# Preprocessing the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# In[22]:


def preprocess_data(data):
    patterns = []
    # Iterate through the dataset entries
    for example in data:
        # Each example contains an 'intents' key
        for intent in example['intents']:
            for pattern in intent['patterns']:
                patterns.append(pattern)
    return patterns


# In[23]:


# Extracting text data from the dataset
text_data = preprocess_data(dataset['train'])
print(dataset)


# In[25]:


get_ipython().system('pip install datasets')
from datasets import Dataset, DatasetDict


# In[26]:


# Create a new dataset for tokenization
new_dataset = {'text': text_data}
new_dataset = DatasetDict({'train': Dataset.from_dict(new_dataset)})


# In[27]:


# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)


# In[28]:


# Tokenize the new dataset
tokenized_dataset = new_dataset['train'].map(tokenize_function, batched=True)


# In[29]:


# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)


# In[30]:


# Trainer: Hugging Face's API for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)


# # Fine tuning the model

# In[ ]:


trainer.train()


# In[ ]:


# Saving the model after training
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

