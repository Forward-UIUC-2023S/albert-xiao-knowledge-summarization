import numpy as np
import torch
import json 
import pandas as pd
from datasets import load_dataset
from transformers.optimization import Adafactor, AdafactorSchedule
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sqlite3
from transformers import BartForConditionalGeneration, BartConfig, BartModel
from transformers import BartTokenizer
from evaluate import load
from sentence_transformers import SentenceTransformer, util
import os
from nltk import tokenize
import nltk
os.environ['HF_HOME'] = '/scratch/huggingface/'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/huggingface/models/'

metric = load("rouge")


import numpy as np
import torch
import json 
import pandas as pd
from datasets import load_dataset
from transformers.optimization import Adafactor, AdafactorSchedule
# from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import list_metrics, load_metric
import requests
import json
from pprint import pprint
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, T5ForConditionalGeneration
_TITLE_LEGNGTH_THRESHOLD = 8
folder = 'checkpoints/'
_DIR = '/scratch/anxiao2/' + folder
_DIR_LOSSES = 'YOUR DIR FOR LOSS/' + folder
_DIR_DATA = '/scratch/ziyic2/'
_DIR_WIKI_PLAYGROUND = '/scratch/ziyic2/wiki_playground/'
from sentence_transformers import SentenceTransformer, util, losses

import random
import os
import time

# _MODEL = 't5-base'
# filename = _DIR + "some.txt"
# os.makedirs(os.path.dirname(filename), exist_ok=True)
# filename = _DIR_LOSSES + "some.txt"
# os.makedirs(os.path.dirname(filename), exist_ok=True)


json_file = _DIR_WIKI_PLAYGROUND + 'wiki_general_question_dataset_03_07_2023_bart.json'
general_question_dataset = []
dataset_file = json_file
with open(dataset_file) as json_data:
    general_question_dataset = json.load(json_data)

train = general_question_dataset['train']
test = general_question_dataset['test']


from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer

device = "cuda:0"
checkpoint = torch.load("/scratch/anxiao2/checkpoints/train_dpr_encoder_20_5500.pt")

#checkpoint = torch.load("/scratch/anxiao2/checkpoints2/train_dpr_encoder_dot0_5000.pt")

tokenizerq = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
modelq = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
tokenizerc = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
modelc = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
modelq.load_state_dict(checkpoint['model_state_dict'])
modelq.eval()
modelc.eval()

from transformers import (
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)
from datasets import Features, Sequence, Value, load_dataset, load_from_disk
from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration, RagConfig, RagModel


_DIR_WIKI = '/scratch/ziyic2/wiki_playground/'
_DIR_WIKI = '/scratch/ziyic2/wiki_dpr_enterprise/'

passages_path = os.path.join(_DIR_WIKI, "enterprise_api_dense_retrieval_dataset")
index_path = os.path.join(_DIR_WIKI, "enterprise_api_dense_retrieval_dataset_hnsw_index.faiss")

true_dataset = load_from_disk(passages_path)
# true_dataset = load_from_disk(passages_path, keep_in_memory=False)
true_dataset.load_faiss_index('embeddings', index_path)
true_dataset.get_index("embeddings")

enterprise_retriever = RagRetriever.from_pretrained(
    'facebook/rag-sequence-nq', index_name="custom", indexed_dataset=true_dataset
)

retriever = RagRetriever.from_pretrained(
    'facebook/rag-sequence-nq', dataset="wiki_dpr", index_name="exact", use_dummy_dataset=True
)
rag_model = RagModel.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
rag_question_encoder = rag_model.question_encoder
# retriever_from_enterprise_api = RagSequenceForGeneration.from_pretrained('facebook/rag-sequence-nq', retriever=enterprise_retriever)
retriever_from_enterprise_api_tokenizer = RagTokenizer.from_pretrained('facebook/rag-sequence-nq')


def retrieve_dpr_enterprise_api(q, n_docs=3, question_encoder=modelq):
    input_ids = retriever_from_enterprise_api_tokenizer(q, return_tensors="pt", padding=True)
    input_ids = input_ids['input_ids']
    embeddings = rag_question_encoder(input_ids)
    res = enterprise_retriever(input_ids, embeddings[0].detach().to(torch.float32).numpy(), n_docs=n_docs)
    # context = []
    outputs_rag = []
    for context_input_ids in res.data['context_input_ids']:
        text = retriever_from_enterprise_api_tokenizer.decode(context_input_ids, skip_special_tokens=True)
        text = text[text.find(' / ') + 3:text.find(' // ')]
        outputs_rag.append(text)
    return '\n'.join(outputs_rag)

def retreival_function(q, n_docs=3, question_encoder=modelq):
    return retrieve_dpr_enterprise_api(q, n_docs, question_encoder)

import time

test_dataloader = DataLoader(test, batch_size=1, shuffle=False)
for j, infos in enumerate(test_dataloader):
    #print(infos)
    
    entities, aspects, summaries = infos
    qs = ['Tell me about {} of {}'.format(aspects[j], entities[j]) for j in range(len(aspects))]
    context = [retreival_function(q, n_docs=3) for q in qs]
    print(context)
    text = qs[0]
    for c in context:
        text += " </s>" + c
    answer = summaries[0]
    scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
    for c in context:
        score = metric.compute(predictions=[c], references=[summaries[0]])
        for key in scores:
            scores[key] += score[key]
    print(score)

    print(text , '\n\n\n', answer, "\n\n\n================================")

    time.sleep(5.0)
    #x_tokens = tokenizer(text, truncation=True, padding=True, max_length=1024, return_tensors="pt").to(device)

    #pred = model.generate(input_ids = x_tokens['input_ids'].to(device))



















def eval(test_dataloader, model, tokenizer):

    num = 0
    scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
    for j, infos in enumerate(test_dataloader):
        num += 1

        entities, aspects, summaries = infos
        qs = ['Tell me about {} of {}'.format(aspects[j], entities[j]) for j in range(len(aspects))]
        context = [retreival_function(q, n_docs=3) for q in qs]
        print(context)
        text = qs[0]
        for c in context:
            text += " </s>" + process(aspects[0] + " " + entities[0], c)
        answer = summaries[0]

        print(text , '\n\n\n', answer)

        x_tokens = tokenizer(text, truncation=True, padding=True, max_length=1024, return_tensors="pt").to(device)

        y_tokens = tokenizer(answer, truncation=True, padding=True, return_tensors="pt").to(device)

        pred = model(input_ids = x_tokens['input_ids'], labels = y_tokens['input_ids'])


        inputs = tokenizer(X,  truncation=True, return_tensors="pt")
        summary_ids = model.generate(inputs["input_ids"].to(device), max_length=100)
        out = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True, max_length=100)[0]
        gt = y
        score = metric.compute(predictions=[out], references=[gt])
        for key in scores:
            scores[key] += score[key]
        if(num < 10):
    
            print(X[0].split('</s>')[0], "\n\n\n*****************************", out, "\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!", gt, score, "\n\n\n==============================")
            #print(X)
        elif(num > 50):
            break
    for key in scores:
        scores[key] /= num      
    print(scores)

