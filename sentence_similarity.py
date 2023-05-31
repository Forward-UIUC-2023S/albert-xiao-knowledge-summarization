import numpy as np
import torch
import json 
import pandas as pd
from datasets import load_dataset
from transformers.optimization import Adafactor, AdafactorSchedule
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import LEDForConditionalGeneration, LEDConfig
from transformers import LEDTokenizer, BertTokenizerFast
import sqlite3
from nltk import tokenize
import nltk
from sentence_transformers import SentenceTransformer, util, losses
import os
import sys

os.environ['HF_HOME'] = '/scratch/huggingface/'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/huggingface/models/'

nltk.download('punkt')
dataset = load_dataset("hotpot_qa", "distractor")

model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
import random


class QADataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.data = dataset
        self.token = tokenizer
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

            
        positive = []
        negative = []
        

        #print(q, title_context, sentences_context, title_index, titles, sent_id)

        for i in range(0, len(titles)):
            #print(i)
            if(sent_id[i] >= len(sentences_context[title_index[titles[i]]])):
                continue
            positive.append(sentences_context[title_index[titles[i]]][sent_id[i]])
        
        
        for i in range(0, len(positive)):
            for j in range(0, 1000):
                rand = random.randint(0, len(title_context) - 1)
                rand2 = random.randint(0, len(sentences_context[rand]) - 1)
                if sentences_context[rand][rand2] not in positive:
                    negative.append(sentences_context[rand][rand2])
                    break
            
            
        return q, positive, negative

train_size = 5000
test_size = 5000
lr = 0.0005
batch_size = 1
num_epochs = 1000

train = dataset['train']

train_trunc = train

test = dataset['validation']

test_trunc = test

train_set = QADataset(train_trunc, tokenizer)
test_set = QADataset(test_trunc, tokenizer)

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

optim = Adafactor(params = model.parameters(), lr = lr, relative_step=False)

class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, q, p, n):
        cosine_scoresp = util.cos_sim(q, p)
        cosine_scoresn = util.cos_sim(q, n)
        return torch.max(((1-cosine_scoresp[0][0]) - (1-cosine_scoresn[0][0]) + 0.25), torch.tensor(0.0))

criterion = CustomLoss()
losses = []
loss_graph = []
for i in range(num_epochs):
    print("starting epoch", i)
    batch_num = 0
    total_loss = 0
    intermed_losses = []
    for q, p, n in train_dataloader:
        #print(q, p, n)
        if(len(p) != len(n)):
            continue
        for j in range(0, len(p)):
            
            pos = p[j][0]
            neg = n[j][0]
            q_embed = model.encode([q[0]], convert_to_tensor=True)
            pos_embed = model.encode([pos], convert_to_tensor=True)
            neg_embed = model.encode([neg], convert_to_tensor=True)
            #print(q_embed, pos_embed, neg_embed)
            optim.zero_grad()
            loss = criterion(q_embed, pos_embed, neg_embed)
            loss.requires_grad = True
            intermed_losses.append(loss.item())
            loss_graph.append(loss.item())
            if(batch_num %100 == 1):  
                print(loss.item())
            loss.backward()
            optim.step()
            total_loss += np.mean(loss.item())
            #progress_bar.update(1)
            
        if batch_num %100 == 0 and batch_num != 0:
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': losses,
            'loss_graph': loss_graph,
            }, '/scratch/anxiao2/checkpoints/train_sentence_encoder_saved_data_' + str(i) + '_' + str(batch_num) + '.pt')
        batch_num += 1
    losses.append(total_loss)