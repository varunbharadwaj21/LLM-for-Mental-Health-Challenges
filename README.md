# LLM Fine-Tuning and Trail Scripts

This repository contains Jupyter Notebooks and converted Python scripts for:
- Fine-tuning language models
- Saving and loading model checkpoints
- Running model interfaces locally
  
🧠 **Project Insight: Emotion Recognition Using Deep Learning**
This project addresses the increasingly important role of emotion recognition in enhancing mental health diagnosis, communication systems, and AI-human interactions. Traditional emotion recognition methods rely on isolated data streams like facial expressions or voice tones, which often fail to grasp the full emotional context. Our project innovates by leveraging deep learning and multimodal data integration to create a scalable and accurate emotion recognition system.

The system is designed to process and analyze text, facial expressions, and voice data to detect emotional states such as happiness, sadness, anxiety, and anger. This multimodal approach provides a more holistic understanding of a person’s emotional profile and offers real-time insights—especially useful in mental health monitoring and therapy support. Unlike many existing models, which operate in silos, this system integrates various data sources to improve reliability and diagnostic power.

#Problem Statement:
Human emotions are subtle, diverse, and context-dependent. Accurately detecting them in real-time requires models that understand not just facial cues or voice modulation, but the interplay of multiple modalities. This system aims to bridge that gap, addressing inconsistencies in data representation, individual variation, and real-time processing challenges.

🌐 Datasets Used
MultiPie Dataset: Facial expression dataset with over 750,000 images across multiple views and lighting.
Mental Health Counseling Conversations: Curated question-answer pairs from online therapy platforms.
Sentiment Analysis for Mental Health: A structured text dataset with emotional context-response pairs.

🔍 Novelty
Multimodal Integration: Simultaneously analyzes text, facial, and vocal inputs.
Real-Time Capability: Designed for continuous emotion monitoring.
Mental Health Focus: Offers diagnostic support to healthcare professionals.

🚀 Future Scope
Use of ViT (Vision Transformers) and BERT/GPT-based text models
Application of Temporal Convolutional Networks (TCNs) and LSTMs for emotion tracking in videos
Development of context-aware systems and culturally inclusive datasets
Real-time deployment on edge-compute devices for privacy-sensitive environments

--
## 💡 Description

The project showcases a complete pipeline of fine-tuning a language model and exploring its capabilities through interactive scripts. While interface code is included, **you must run it locally** to see the full outputs.

> ⚠️ **Note:** Some outputs are not visible due to notebook execution not being saved before upload. Please re-run the notebooks for full results.

## 📁 Files Included

- `llmtrail.py` – Script converted from `llmtrail.ipynb`
- `finetuning and saving model.py` – Script converted from `finetuning and saving model.ipynb`
- `README.md` – Project overview and licensing information

## 🚀 Run Instructions

1. Clone the repository
2. Set up a Python environment (recommended: Python 3.8+ with `transformers`, `datasets`, `torch`)
3. Run the `.py` files or Jupyter notebooks locally for best experience

```bash
pip install transformers datasets torch
python "finetuning and saving model.py"
```

## 🔒 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**.

### You are free to:
- **Share** — copy and redistribute the material in any medium or format

### Under the following terms:
- **Attribution** — You must give appropriate credit.
- **NonCommercial** — You may not use the material for commercial purposes.
- **NoDerivatives** — If you remix, transform, or build upon the material, you may not distribute the modified material.

> 🚫 Commercial use and redistribution with changes are not allowed. Citation is mandatory.

## 🧠 Citation

If you use any part of this codebase, please cite this work and mention the author in your project.
