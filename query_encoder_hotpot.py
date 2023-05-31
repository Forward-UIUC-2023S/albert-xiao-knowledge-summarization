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
from sentence_transformers import SentenceTransformer, util, losses

import random
import os
import time



from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer

device = "cuda:2"

tokenizerq = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
modelq = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
tokenizerc = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
modelc = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)


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

dataset = load_dataset("hotpot_qa", "distractor")

class QADataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        
        q = data['question']
        
        context = data['context']
        title_context = context['title']
        sentences_context = context['sentences']
        
        evidence = data['supporting_facts']
        titles = evidence['title']
        sent_id = evidence['sent_id']
        
        
        title_index = {}

        for i in range(0, len(title_context)):
            title_index[title_context[i]] = i

            
        positive = ""        

        for i in range(0, len(titles)):
            #print(i)
            if(sent_id[i] >= len(sentences_context[title_index[titles[i]]])):
                continue
            positive += sentences_context[title_index[titles[i]]][sent_id[i]] + " "
        
        return q, positive

train = dataset['train']
train_set = QADataset(train)
train_dataloader = DataLoader(train_set, batch_size=16, shuffle=False)

for i in range(25):
    batch_num = 0
    total_loss = 0
    intermed_losses = []
    for q, p in train_dataloader:
        #print(infos)
        
    
        qembed = modelq(tokenizerq(q, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)).pooler_output
        cembed = modelc(tokenizerc(p, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)).pooler_output
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
            }, '/scratch/anxiao2/checkpoints2/train_dpr_encoder_hotpot' + str(i) + '_' + str(batch_num-1) + '.pt')
        batch_num += 1
    losses.append(total_loss)
