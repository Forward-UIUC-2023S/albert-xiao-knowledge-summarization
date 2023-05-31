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

device = 'cuda:0'
tokenizerq = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
modelq = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
tokenizerc = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
modelc = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)


import requests
def retreival_function(q, n_docs):
    url = 'http://127.0.0.1:5000/retreive_from_enterprise_api'
    res = requests.get(url, params={'query': q, 'k':n_docs})
    passages = res.json()['res']
    return passages


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, m):
        m = torch.exp(m)
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
        context = [retreival_function(q, n_docs=1) for q in qs]
        qembed = modelq(tokenizerq(qs, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)).pooler_output
        cembed = modelc(tokenizerc(context, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)).pooler_output
        sim = qembed @ cembed.T 
        optim.zero_grad()
        loss = criterion(sim)
        #loss.requires_grad = True
        intermed_losses.append(loss.item())
        loss_graph.append(loss.item())
        #print(loss.item())
        if batch_num %1000 == 1:
            print(loss.item())
            torch.save({
            'epoch': i,
            'model_state_dict': modelq.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': losses,
            'loss_graph': loss_graph,
            }, '/scratch/anxiao2/checkpoints2/train_bart_on_dataset' + str(i) + '_' + str(batch_num-1) + '.pt')
        batch_num += 1
    losses.append(total_loss)
