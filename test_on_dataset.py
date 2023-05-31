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
import requests
os.environ['HF_HOME'] = '/scratch/huggingface/'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/huggingface/models/'
nltk.download('punkt')
checkpoint = torch.load('/scratch/anxiao2/checkpoints/train_sentence_encoder_saved_data_0_84000.pt')

_TITLE_LEGNGTH_THRESHOLD = 8
folder = 'checkpoints/'
_DIR = '/scratch/anxiao2/' + folder
_DIR_LOSSES = 'YOUR DIR FOR LOSS/' + folder
_DIR_DATA = '/scratch/ziyic2/'
_DIR_WIKI_PLAYGROUND = '/scratch/ziyic2/wiki_playground/'

json_file = _DIR_WIKI_PLAYGROUND + 'wiki_general_question_dataset_03_07_2023_bart.json'
general_question_dataset = []
dataset_file = json_file
with open(dataset_file) as json_data:
    general_question_dataset = json.load(json_data)

train = general_question_dataset['train']
test = general_question_dataset['test']

sent_model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')
sent_model.load_state_dict(checkpoint['model_state_dict'])

metric = load("rouge")


tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

config = BartConfig()
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")


lr = 0.0005
batch_size = 1
num_epochs = 10


device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
optim = Adafactor(params = model.parameters(), lr = lr, relative_step=False)

def process(query, text):
    sentences = tokenize.sent_tokenize(text)
    if(len(sentences) == 0):
        return ''
    query_embed = sent_model.encode([query], convert_to_tensor=True)
    doc_embed = sent_model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embed, doc_embed)
    top5 = torch.topk(cosine_scores[0], min(1, len(cosine_scores[0])), sorted=False)[1]
    out = ''
    for i in top5:
        out += sentences[i] + ' '
    return out

def retreival_function(q, n_docs):
    url = 'http://127.0.0.1:5000/retreive_from_enterprise_api'
    res = requests.get(url, params={'query': q, 'k':n_docs})
    passages = res.json()['res']
    return passages
loss_graph = []
losses = []

train_dataloader = DataLoader(train, batch_size=1, shuffle=True)
for i in range(25):
    batch_num = 0
    total_loss = 0
    intermed_losses = []
    for j, infos in enumerate(train_dataloader):
        #print(infos)
        
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

        optim.zero_grad()
        loss = pred.loss
        intermed_losses.append(loss.item())
        loss_graph.append(loss.item())
        loss.backward()
        optim.step()
        total_loss += np.mean(loss.item())
        if batch_num %100 == 1:
            print(loss.item())
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': losses,
            'loss_graph': loss_graph,
            }, '/scratch/anxiao2/checkpoints2/train_model_on_dataset' + str(i) + '_' + str(batch_num-1) + '.pt')
        batch_num += 1
    losses.append(total_loss)
