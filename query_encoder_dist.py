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

device = "cuda:2"

tokenizerq = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
modelq = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
tokenizerc = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
modelc = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)



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


def retrieve_dpr_enterprise_api(q, n_docs, question_encoder):
    input_ids = retriever_from_enterprise_api_tokenizer(q, return_tensors="pt", padding=True)
    input_ids = input_ids['input_ids']
    embeddings = question_encoder(input_ids.to(device))
    res = enterprise_retriever(input_ids, embeddings[0].detach().to(torch.float32).to('cpu').numpy(), n_docs=n_docs)
    # context = []
    outputs_rag = []
    for context_input_ids in res.data['context_input_ids']:
        text = retriever_from_enterprise_api_tokenizer.decode(context_input_ids, skip_special_tokens=True)
        text = text[text.find(' / ') + 3:text.find(' // ')]
        outputs_rag.append(text)
    return '\n'.join(outputs_rag)



import requests
def retreival_function(q, n_docs, question_encoder):
    return retrieve_dpr_enterprise_api(q, n_docs, question_encoder)



class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, q, p):
        m = torch.exp(torch.cdist(p, q))
        pos = m[torch.arange(len(m)), torch.arange(len(m))]
        neg = torch.sum(m, 1)
        loss = torch.sum(-torch.log(pos/neg))
        return loss

losses = []
loss_graph = []
modelq.train()
lr = 0.0003
criterion = CustomLoss()
optim = Adafactor(params = modelq.parameters(), lr = lr, relative_step=False)



train_dataloader = DataLoader(train, batch_size=16, shuffle=True)
for i in range(25):
    batch_num = 0
    total_loss = 0
    intermed_losses = []
    for j, infos in enumerate(train_dataloader):
        #print(infos)
        
        entities, aspects, summaries = infos
        qs = ['Tell me about {} of {}'.format(aspects[j], entities[j]) for j in range(len(aspects))]
        context = [retreival_function(q, 1, modelq) for q in qs]
        qembed = modelq(tokenizerq(qs, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)).pooler_output
        cembed = modelc(tokenizerc(context, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)).pooler_output
        optim.zero_grad()
        loss = criterion(qembed, cembed)
        #loss.requires_grad = True
        intermed_losses.append(loss.item())
        loss_graph.append(loss.item())
        #print(loss.item())
        if batch_num %100 == 1:
            print(loss.item())
            torch.save({
            'epoch': i,
            'model_state_dict': modelq.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': losses,
            'loss_graph': loss_graph,
            }, '/scratch/anxiao2/checkpoints2/train_dpr_encoder_cdist' + str(i) + '_' + str(batch_num-1) + '.pt')
        batch_num += 1
    losses.append(total_loss)
