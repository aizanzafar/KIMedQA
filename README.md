# KIMedQA: Towards Building Knowledge-Enhanced Medical QA Models

[![DOI:10.1007/s10844-024-00844-1](https://img.shields.io/badge/DOI-10.1007/s10844--024--00844--1-blue)](https://doi.org/10.1007/s10844-024-00844-1)

## Description

KIMedQA is a knowledge-infused medical question-answering (QA) system designed to address challenges in extracting accurate and comprehensive answers in the medical domain. It employs relevant knowledge graph (KG) selection, pruning of large-scale graphs, and integration with pre-trained language models to enhance the accuracy and relevance of medical QA.

This repository provides the code, datasets, and instructions to reproduce the results published in:

**Zafar, A., Sahoo, S. K., Varshney, D., Das, A., Ekbal, A. (2024).**  
*[KIMedQA: Towards Building Knowledge-Enhanced Medical QA Models](https://doi.org/10.1007/s10844-024-00844-1).*  
Journal of Intelligent Information Systems, 62, 833–858.

---

## Features

- Knowledge Graph construction and pruning for relevant medical entities.
- Contextual representation using RoBERTa-large for question-answering tasks.
- Benchmarks on MASH-QA and COVID-QA datasets.
- Empirical evaluation comparing KIMedQA with state-of-the-art models and ChatGPT.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aizanzafar/KIMedQA.git
   cd KIMedQA
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Data Preparation

Ensure datasets (`train.json`, `val.json`, and `test.json`) are available in the `Data/` folder. A sample dataset is provided.

### 2. Knowledge Graph Construction

#### Build KG using UMLS
Navigate to the `UMLS_KG_Preprocess/` folder and execute:
```bash
python final_preprocess.py
```

### 3. Preprocessing

Preprocess the dataset for QA tasks:
```bash
python preprocess.py
```

### 4. Training the Model

Train the KIMedQA model:
```bash
python qa_st.py
```

#### Default Training Arguments:
```python
train_args = {
    'learning_rate': 2e-5,
    'num_train_epochs': 3,
    'max_seq_length': 512,
    'doc_stride': 384,
    'output_dir': "outputs/",
    'overwrite_output_dir': True,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 8,
}
model = QuestionAnsweringModel("roberta", "roberta-large", args=train_args)
```
---

## Datasets

### MASH-QA
- Source: [MASH-QA GitHub](https://github.com/mingzhu0527/MASHQA)
- Description: Consumer healthcare questions from WebMD.

### COVID-QA
- Source: [COVID-QA Dataset](https://github.com/deepset-ai/COVID-QA)
- Description: QA dataset based on scholarly research on COVID-19.

---

## Reproducibility

1. Follow the installation and data preparation steps.
2. Train the model using the provided scripts.
3. Evaluate using the default configurations or adjust hyperparameters as needed.

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{zafar2024kimedqa,
  title={KIMedQA: Towards Building Knowledge-Enhanced Medical QA Models},
  author={Zafar, Aizan and Sahoo, Sovan Kumar and Varshney, Deeksha and Das, Amitava and Ekbal, Asif},
  journal={Journal of Intelligent Information Systems},
  volume={62},
  pages={833--858},
  year={2024},
  publisher={Springer}
}
```

---

## Acknowledgments

This research is supported by the projects "Percuro: A Holistic Solution for Text Mining" (Wipro Ltd.) and "Sevak: An Intelligent Indian Language Chatbot" (SERB, Government of India).
