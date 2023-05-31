
# <Knowledge Summarization: Query Focused Summarization Component>

## Overview

This module explores mutliple ways of training a Query Focused Summarization module as part of a Retrieval Augmented Generation Pipeline, including a base model using LED, evidence selection and BART, and coreference resolution. This module also contains some WIP code for training the DPR encoders for the RAG pipeline.

## Setup

Installing Dependencies: 

1. Python 3.8.10 and pip 20.0.2

2. Install dependencies with requirements.txt
```
pip install -r requirements.txt 
```




```
albert-xiao-knowledge-summarization/
    - c4.db
    - requirements.txt
    - dpr_eval.py
    - dpr_server.py
    - query_encoder_cont.py
    - query_encoder_cos.py
    - query_encoder_dist.py
    - query_encoder_hotpot.py
    - query_encoder.py
    - sentence_similarity.py
    - similarity.py
    - test_from_file_bart_coref_top3.py
    - test_from_file_bart_coref.py
    - test_from_file_bart_no_evidence.py
    - test_from_file_bart_top3.py
    - test_from_file_bart.py
    - test_from_file.py
    - test_on_dataset_control.py
    - test_on_dataset.py
    - test_sent_fine_tune_bart.py
    - train_sent_bart_0_1500.pt
    - train_sentence_encoder_saved_data_0_84000.pt
    - train_similarity_saved_data_mar7_0_1500.pt
    - train.py

```

- c4.db - C4 cleaned dataset as an SQLite database file for easy loading
- requirements.txt - python library dependencies
- dpr_eval.py - WIP DPR evaluation script
- dpr_server.py - DPR server to serve retrieved documents
- query_encoder_cont.py - WIP training loop for DPR query encoding with contrastive loss function
- query_encoder_cos.py - WIP training loop for DPR query encoding with cosine distance loss function
- query_encoder_dist.py - WIP training loop for DPR query encoding with contrastive loss function
- query_encoder_hotpot.py - WIP training loop for DPR query encoding end to end with HotpotQA downstream task
- query_encoder.py - WIP training loop for DPR query encoding from paper
- sentence_similarity.py - fine tuning loop with contrastive loss function for BERT sentence transformer
- similarity.py - training loop for LED model
- test_from_file_bart_coref_top3.py - evaluate BART model with top 3 evidence selection with pronoun replacement from checkpoint file
- test_from_file_bart_coref.py - evaluate BART model with top 1 evidence selection with pronoun replacement from checkpoint file
- test_from_file_bart_no_evidence.py - evaluate BART model with no evidence selection from checkpoint file
- test_from_file_bart_top3.py - evaluate BART model with top 3 evidence selection from checkpoint file 
- test_from_file_bart.py - evaluate BART model with top 1 evidence selection from checkpoint file
- test_from_file.py - evaluate LED model from checkpoint file
- test_on_dataset_control.py - WIP fine tuning QFS model to Ziyi's WikiASTL dataset
- test_on_dataset.py - WIP fine tuning QFS model to Ziyi's WikiASTL dataset
- test_sent_fine_tune_bart.py - fine-tuning loop for bart with top k evidence selection
- train_sent_bart_0_1500.pt - fine-tuned bart checkpoint, 1500 datapoints
- train_sentence_encoder_saved_data_0_84000.pt - fine-tuned sentence encoder, 84000 datapoints
- train_similarity_saved_data_mar7_0_1500.pt - fine-funed LED checkpoint, 1500 datapoints
- train.py - training loop for LED basic model


## Functional Design (Usage)

All python files are scripts. They are meant to be simply run.

## Demo video


## Algorithmic Design 

Basic model: fine-tune longformer LED model by passing in all documents and query separated by SEP tokens
Evidence selection: fine-tune BERT sentence transformer using HotpotQA dataset and contrastive loss function to encode query-evidence pairs such that they have high cosine similarity. Then select top k sentences of each document and query as input to fine-tune BART QFS model
Coreference Resolution: perform pronoun replacement before

## Issues and Future Work

Training the RAG encoders is currently WIP
Train the QFS models for longer and tune hyperparameters

## Change log

Spring 2023 (Albert Xiao)
* Created this module

## References 

Datasets: AQuaMuSe https://huggingface.co/datasets/aquamuse 
          HotpotQA https://hotpotqa.github.io/ 

Papers: Longformer https://arxiv.org/abs/2004.05150 
        Retrieval Augmented Generation https://arxiv.org/abs/2005.11401
        Dense Passage Retrieval https://arxiv.org/abs/2004.04906
        BART https://arxiv.org/abs/1910.13461
        Exploring Nueral Methods for QFS https://arxiv.org/abs/2112.07637
