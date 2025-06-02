#!/usr/bin/env python
# coding: utf-8

# In[1]:
from datasets import load_dataset

ds1 = load_dataset("Amod/mental_health_counseling_conversations", split="train")
ds2 = load_dataset("heliosbrahma/mental_health_chatbot_dataset", split="train")
ds3 = load_dataset("ZahrizhalAli/mental_health_conversational_dataset", split="train")

print("🧾 Dataset 1 Columns:", ds1.column_names)
print("🧾 Dataset 2 Columns:", ds2.column_names)
print("🧾 Dataset 3 Columns:", ds3.column_names)


# In[2]:

from datasets import load_dataset
import pandas as pd

# Load datasets
print("Loading...")
ds1 = load_dataset("Amod/mental_health_counseling_conversations", split="train")
ds2 = load_dataset("heliosbrahma/mental_health_chatbot_dataset", split="train")
ds3 = load_dataset("ZahrizhalAli/mental_health_conversational_dataset", split="train")

# Process dataset 1 directly
df1 = pd.DataFrame({
    "instruction": ds1["Context"],
    "response": ds1["Response"]
})

# Dataset 2 and 3 have 'text' column with Q/A inside
def extract_qa(text):
    parts = text.split("A:")
    question = parts[0].replace("Q:", "").strip() if len(parts) > 1 else ""
    answer = parts[1].strip() if len(parts) > 1 else ""
    return pd.Series([question, answer])

# Process dataset 2
df2_raw = pd.DataFrame(ds2)
df2_qa = df2_raw['text'].apply(extract_qa)
df2_qa.columns = ["instruction", "response"]

# Process dataset 3
df3_raw = pd.DataFrame(ds3)
df3_qa = df3_raw['text'].apply(extract_qa)
df3_qa.columns = ["instruction", "response"]

# Combine and clean
combined_df = pd.concat([df1, df2_qa, df3_qa], ignore_index=True)
combined_df.dropna(inplace=True)
combined_df.drop_duplicates(inplace=True)

# Save
combined_df.to_json("mental_health_combined.json", orient="records", lines=True)
print(f"Done! Saved {len(combined_df)} cleaned examples to 'mental_health_combined.json'")


# In[3]:


import json

# Load JSONL dataset
file_path = "mental_health_combined.json"
dataset = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        dataset.append(json.loads(line))

print(f"Loaded {len(dataset)} entries.")


# In[4]:
import json

# Load your merged dataset
with open("mental_health_combined.json", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Convert to training text (GPT-2 expects plain text)
with open("gpt2_training.txt", "w", encoding="utf-8") as out:
    for line in lines:
        item = json.loads(line)
        prompt = f"Instruction: {item['instruction']}\nResponse: {item['response']}\n"
        out.write(prompt + "\n")


# In[5]:


from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# In[6]:


pip install huggingface_hub[hf_xet]


# In[7]:


# Load tokenizer and model
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))


# In[8]:


# Prepare dataset
train_path = "gpt2_training.txt"
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# In[9]:


pip install transformers[torch]


# In[10]:


pip install accelerate>=0.26.0


# In[11]:


# Training configuration
training_args = TrainingArguments(
    output_dir="./gpt2_mentalhealth_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir="./logs",
    logging_steps=100
)


# In[13]:


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Start training
trainer.train()


# In[ ]:


# Save the fine-tuned model
model.save_pretrained("./gpt2_mentalhealth_model")
tokenizer.save_pretrained("./gpt2_mentalhealth_model")
print("Model saved to ./gpt2_mentalhealth_model")


# In[ ]:


from tomlkit import dump


# In[ ]:


model.save_pretrained("./gpt2_mentalhealth_model")
tokenizer.save_pretrained("./gpt2_mentalhealth_model")


# In[ ]:


import os
os.listdir("gpt2_mentalhealth_model")


# In[ ]:


import os
print(os.path.exists("./gpt2_mentalhealth_model"))
print(os.listdir("./gpt2_mentalhealth_model"))


# In[ ]:


from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from pathlib import Path

model_dir = Path("./gpt2_mentalhealth_model")

# Load model config
config = GPT2Config.from_json_file(str(model_dir / "config.json"))

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(
    model_dir,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

# Load model from safetensors
model = GPT2LMHeadModel.from_pretrained(
    model_dir,
    config=config,
    local_files_only=True,
    use_safetensors=True
)


# In[ ]:


#  Load fine-tuned GPT2 model and tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained(
    "./gpt2_mentalhealth_model",
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(
    "./gpt2_mentalhealth_model",
    local_files_only=True,
    use_safetensors=True
)


# In[ ]:


from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr

# Load emotion classifier
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Force local loading of tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(
    "./gpt2_mentalhealth_model",
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

# Load the model using safetensors
model = GPT2LMHeadModel.from_pretrained(
    "./gpt2_mentalhealth_model",
    local_files_only=True,
    use_safetensors=True
)


# In[ ]:


def detect_emotions(text):
    emotions = emotion_classifier(text)
    return emotions


# In[ ]:


def generate_response(user_input, emotion_hint):
    prompt = f"The user seems to be feeling {emotion_hint}.\nMessage: {user_input}\nResponse:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens).replace(prompt, "")


# In[ ]:


def mental_health_chat(user_input):
    emotions = detect_emotions(user_input)
    emotion_hint = emotions[0]['label']
    emotion_summary = ", ".join([f"{e['label']} ({e['score']:.2f})" for e in emotions])
    response = generate_response(user_input, emotion_hint)
    return response.strip(), f"Detected emotions: {emotion_summary}"


# In[ ]:


import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from pathlib import Path

# Load model & tokenizer
model_dir = Path("./gpt2_mentalhealth_model")
config = GPT2Config.from_json_file(str(model_dir / "config.json"))

tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(
    model_dir, config=config, local_files_only=True, use_safetensors=True
)

# Chat function
def mental_health_chat(user_input):
    prompt = f"The user says: {user_input}\nResponse:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=150,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

# Launch Gradio app
gr.Interface(
    fn=mental_health_chat,
    inputs=gr.Textbox(lines=3, placeholder="How are you feeling today?"),
    outputs="text",
    title="Mental Health LLM Companion",
    description="Your AI-based support listener."
).launch()


# In[ ]:


import gradio as gr
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    pipeline
)
from pathlib import Path

# === Load Emotion Detection Model ===
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=3,
    framework="pt"
)

# === Load Fine-Tuned GPT2 Model ===
model_dir = Path("./gpt2_mentalhealth_model")
config = GPT2Config.from_json_file(str(model_dir / "config.json"))

tokenizer = GPT2Tokenizer.from_pretrained(
    model_dir,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(
    model_dir,
    config=config,
    local_files_only=True,
    use_safetensors=True
)

# === Chat Function with Emotion Detection ===
def mental_health_chat(user_input):
    try:
        # Emotion Detection
        emotions = emotion_classifier(user_input)

        # Flatten list if nested
        if isinstance(emotions[0], list):
            emotions = emotions[0]

        emotion_hint = emotions[0]['label']
        emotion_summary = "\n".join([f"- {e['label']} ({e['score']:.2f})" for e in emotions])

        # GPT-2 Generation with sampling strategies
        prompt = f"The user seems to be feeling {emotion_hint}.\nMessage: {user_input}\nResponse:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=150,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,       # Prevent repeating n-grams
            repetition_penalty=1.2,       # Penalize repeating words
            temperature=0.9,              # Encourage variety
            top_k=50,                     # Sample from top 50 tokens
            top_p=0.95,                   # Nucleus sampling
            do_sample=True                # Enable randomness
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()

        return response, f"💬 Detected Emotion: **{emotion_hint}**\n\n🔍 Top Predictions:\n{emotion_summary}"

    except Exception as e:
        return f"Error: {str(e)}", "⚠️ Emotion detection failed"

# === Launch Gradio App ===
gr.Interface(
    fn=mental_health_chat,
    inputs=gr.Textbox(
        lines=3,
        placeholder="How are you feeling today?",
        label="🗣️ Tell me what’s on your mind"
    ),
    outputs=[
        gr.Textbox(label="AI Response"),
        gr.Textbox(label="Emotion Analysis")
    ],
    title="Mental Health LLM Companion",
    description="This tool provides empathetic AI responses and detects emotional tone. Not a substitute for professional help."
).launch()


# In[ ]:


import gradio as gr
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, pipeline
)
from pathlib import Path
from collections import Counter

# Load emotion detection pipeline
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=3,
    framework="pt"
)

# Load fine-tuned GPT-2 model
model_dir = Path("./gpt2_mentalhealth_model")
config = GPT2Config.from_json_file(str(model_dir / "config.json"))
tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(
    model_dir, config=config, local_files_only=True, use_safetensors=True
)


# In[ ]:


# Helper to filter hallucinated names or excessive repetition
def clean_response(text):
    lines = [line for line in text.split('\n') if 'sherry' not in line.lower()]
    sentences = [s.strip() for s in ' '.join(lines).split('.') if s.strip()]
    counter = Counter(sentences)
    filtered = [s for s in sentences if counter[s] < 3]
    return '. '.join(filtered) + '.' if filtered else 'Let’s take a moment to reflect together.'


# In[ ]:


# Main chat function
def mental_health_chat(user_input):
    try:
        emotions = emotion_classifier(user_input)
        if isinstance(emotions[0], list):
            emotions = emotions[0]
        emotion_hint = emotions[0]['label']
        emotion_summary = '\n'.join([f"- {e['label']} ({e['score']:.2f})" for e in emotions])

        prompt = (
            f"The user is experiencing {emotion_hint} and says: '{user_input}'."
            f" Provide an empathetic response without mentioning any names.\nResponse:"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=150,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=4,
            repetition_penalty=1.3,
            temperature=0.85,
            top_k=50,
            top_p=0.92,
            do_sample=True
        )
        raw_response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = raw_response.replace(prompt, '').strip()
        return clean_response(response), f"💬 Detected Emotion: **{emotion_hint}**\n\n🔍 Top Predictions:\n{emotion_summary}"
    except Exception as e:
        return f" Error: {str(e)}", "⚠️ Emotion detection failed"


# In[ ]:


# Gradio interface
gr.Interface(
    fn=mental_health_chat,
    inputs=gr.Textbox(lines=3, label="🗣️ Share your thoughts"),
    outputs=[
        gr.Textbox(label="🧠 AI Response"),
        gr.Textbox(label="🎭 Emotion Analysis")
    ],
    title="🧠 Mental Health LLM Companion",
    description="Empathetic AI + Emotion detection. No hallucinated people. Just thoughtful support."
).launch()

