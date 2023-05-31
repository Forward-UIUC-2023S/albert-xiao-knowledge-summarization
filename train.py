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
from transformers import LEDTokenizer
import sqlite3


aquamuse = load_dataset("aquamuse", "abstractive")


tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")

config = LEDConfig()
#model = LEDModel(config)
model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")

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

    def __init__(self, dataset, tokenizer, db):
        self.data = dataset
        self.token = tokenizer
        self.db = db
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        docs = self.data[idx]['query']
        for url in self.data[idx]['input_urls']:
            text = self.db.query(url)[0]
            docs += text + "</s>"
        docs = docs[:-4]
        return docs, self.data[idx]['target']
        
    
    
dataset = aquamuse

DB_PATH = "/home/anxiao2/knowledge_summarization/c4.db"
DATA_PATH = "/disk2/c4/c4/en.noblocklist/"
db = AquaMuse(DB_PATH)

train = dataset['train']

train_trunc = np.array([train[i] for i in range(0, train_size)])

test = dataset['test']

test_trunc = np.array([test[i] for i in range(0, test_size)])

train_set = SummDataset(train_trunc, tokenizer, db)
test_set = SummDataset(test_trunc, tokenizer, db)

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


progress_bar = tqdm(range(num_epochs*train_size))

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

model.to(device)
optim = Adafactor(params = model.parameters(), lr = lr, relative_step=False)

losses = []
loss_graph = []
for i in range(num_epochs):
    print("starting epoch", i)
    batch_num = 0
    total_loss = 0
    intermed_losses = []
    for X, y in train_dataloader:
        batch_num += 1
        x_tokens = tokenizer(X, truncation=True, padding="longest", return_tensors="pt").to(device)

        y_tokens = tokenizer(y, truncation=True, padding="longest", return_tensors="pt").to(device)

        #pred = model.forward(input_ids = x_tokens['input_ids'], attention_mask=x_tokens['attention_mask'], decoder_input_ids = y_tokens['input_ids'], decoder_attention_mask=y_tokens['attention_mask'])
        #pred = model.forward(input_ids = x_tokens['input_ids'], attention_mask=x_tokens['attention_mask'], labels = y_tokens['input_ids'])
        pred = model.forward(input_ids = x_tokens['input_ids'], labels = y_tokens['input_ids'])
        print("forward")
          

        optim.zero_grad()
        loss = pred.loss
        intermed_losses.append(loss.item())
        loss_graph.append(loss.item())
        if(batch_num == 1):  
          print(loss.item())
        loss.backward()
        optim.step()
        total_loss += np.mean(loss.item())
        #progress_bar.update(1)
        
        if batch_num %100 == 1:
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': losses,
            'loss_graph': loss_graph,
            }, '/scratch/anxiao2/checkpoints/pretrain_saved_data2_' + str(i) + '_' + str(batch_num-1) + '.pt')

    losses.append(total_loss)

    
