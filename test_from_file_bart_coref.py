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
import neuralcoref
import spacy
nlp = spacy.load('en')
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')



nltk.download('punkt')

checkpoint = torch.load('/scratch/anxiao2/checkpoints/train_sentence_encoder_saved_data_0_84000.pt')

os.environ['HF_HOME'] = '/scratch/huggingface/'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/huggingface/models/'
sent_model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')
sent_model.load_state_dict(checkpoint['model_state_dict'])

metric = load("rouge")

aquamuse = load_dataset("aquamuse", "abstractive")


tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

config = BartConfig()
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

train_size = len(aquamuse['train'])
test_size = len(aquamuse['test'])
lr = 0.0005
batch_size = 1
num_epochs = 1000



class AquaMuse():
    def __init__(self, DB_PATH):
        self.DB_PATH = DB_PATH
        self.conn = sqlite3.connect(self.DB_PATH)
        self.cur = self.conn.cursor()
    def query(self, url):
        self.cur.execute("SELECT data FROM aquamuse WHERE url=?", (url,))
        fetch = self.cur.fetchall()
        if(len(fetch) == 0):
            return ("",)
        else:
            return fetch[0]
    def close(self):
        self.conn.close()
        self.cur.close()


class SummDataset(Dataset):

    def __init__(self, dataset, tokenizer, db, sent_model):
        self.data = dataset
        self.token = tokenizer
        self.db = db
        self.sent_model = sent_model
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q = self.data[idx]['query']
        docs = q
        for url in self.data[idx]['input_urls']:
            text = self.db.query(url)[0]
            best = self.process(q, text)
            docs += "</s>" + best 
        return docs, self.data[idx]['target']
        
    def process(self, query, text):
        sentences = tokenize.sent_tokenize(text)
        for i in range(0, len(sentences)-1):
            twosent = sentences[i] + " " + sentences[i+1]
            doc = nlp(twosent)
            if doc._.has_coref:
                #print(tokenize.sent_tokenize(doc._.coref_resolved))
                newsents = tokenize.sent_tokenize(doc._.coref_resolved)
                if len(newsents) > 1:
                    sentences[i+1] = newsents[1]
                else:
                    sentences[i+1] = newsents[0]
        if(len(sentences) == 0):
            return ''
        query_embed = self.sent_model.encode([query], convert_to_tensor=True)
        doc_embed = self.sent_model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embed, doc_embed)
        top5 = torch.topk(cosine_scores[0], min(1, len(cosine_scores[0])), sorted=False)[1]
        out = ''
        for i in top5:
           out += sentences[i] + ' '
        return out
        
    
    
dataset = aquamuse

DB_PATH = "/home/anxiao2/knowledge_summarization/c4.db"
DATA_PATH = "/disk2/c4/c4/en.noblocklist/"
db = AquaMuse(DB_PATH)

test = dataset['test']

test_trunc = np.array([test[i] for i in range(0, test_size)])

test_set = SummDataset(test_trunc, tokenizer, db, sent_model)

test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

device = "cuda:1" if torch.cuda.is_available() else "cpu"

model.to(device)

checkpoint = torch.load('/scratch/anxiao2/checkpoints/train_sent_bart_0_1000.pt')
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
model.load_state_dict(checkpoint['model_state_dict'])
optim = Adafactor(params = model.parameters(), lr = lr, relative_step=False)
optim.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

num = 0
scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
for X, y in test_dataloader:
    num += 1

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
