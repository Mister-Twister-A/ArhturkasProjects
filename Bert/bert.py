import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy
import torchtext
from torchtext.vocab import GloVe


from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import string
import itertools
import random
import re
from collections import defaultdict, Counter
import math



with open("tiny-shakespeare.txt", "r") as f:
    text_DATA = f.read()

#cleaning the data
sentances_data = re.sub(r"[\n,\-'&;]", ' ', text_DATA.lower())
sentances_data = re.split('\.|:|\!|\?', sentances_data)

sentances_list = [s.split() for s in sentances_data] 

word_list = list(sorted(set(''.join(sentances_data).split())))
print(len(word_list))

#making dict to convert from string to index and index to string
word_to_int = {"[PAD]" : 0, "[CLS]" : 1, "[SEP]": 2, "[MASK]": 3}
for i,word in enumerate(word_list):
    word_to_int[word] = i + 4
int_to_word = {i:w for i,w in enumerate(word_list)}
print(len(word_to_int.keys()))
vocab_size = len(word_to_int.keys()) + 4
batch_size = 4

class TweetDataset(Dataset):

    def __init__(self, mask_percent, sentances, optimal_pad_percent):
        self.mask_percent = mask_percent
        self.optimal_pad_percent = optimal_pad_percent
        self.sentances = sentances

        lens = np.array([len(s) for s in self.sentances]) 
        self.sentance_len = int(np.percentile(lens, self.optimal_pad_percent)) # getting the otimal length of the sentance to pad
        #print(self.sentance_len)
        
        _word_list = list(sorted(set(''.join(sentances_data).split())))
        self.vocab_size = len(_word_list)
        
        self.word_to_int = {"[PAD]" : 0, "[CLS]" : 1, "[SEP]": 2, "[MASK]": 3} #making dict to convert from string to index and index to string
        for i,word in enumerate(_word_list):
            self.word_to_int[word] = i + 4
        self.int_to_word = {i:w for i,w in enumerate(_word_list)}
        

    def _pad_sequence(self, seq, itmask=[]): # itmask = inverse token mask
        len_s = len(seq)
        pad_seq = []
        pad_itmask = []

        if len_s >= self.sentance_len:
            pad_seq = seq[:self.sentance_len]
        else:
            pad_seq = seq + ["[PAD]"] * (self.sentance_len - len_s)
        
        len_m = len(itmask)
        if len_m >= self.sentance_len:
            pad_itmask = itmask[:self.sentance_len]
        else:
            pad_itmask = itmask + [True] * (self.sentance_len - len_m)

        return pad_seq, pad_itmask
    
    def _mask_seq(self, seq):
        len_s = len(seq)
        itmask = [False for i in range(len_s)] # inverse token mask
        

        mask_amount = round(self.mask_percent * len_s)
        for i in range(mask_amount):
            idx = random.randint(0, len_s-1)

            if random.random() < 0.8:
                seq[idx] = "[MASK]"
            else:
                seq[idx] = self.int_to_word[random.randint(4, self.vocab_size-1)]
            itmask[idx] = True
        return seq, itmask
    
    def _preprocess_sentance(self, sentance):
        mask, itmask = self._mask_seq(sentance)
        pad, itpad = self._pad_sequence(mask, itmask=itmask)
        return pad, itpad

    
    def __len__(self, ):
        return len(self.sentances)

    def __getitem__(self, idx):
        nsp_tgt = random.randint(0,1)
        sentance_a = self.sentances[idx]
        sentence_b=[]
        if nsp_tgt == 1 and idx != len(self.sentances)-1:
            sentence_b=self.sentances[idx+1]
        elif idx != len(self.sentances)-1:
            sentence_b = self.sentances[random.randint(0, len(self.sentances)-1)]
        
        prep_a, itm_a = self._preprocess_sentance(sentance_a)
        prep_b, itm_b = self._preprocess_sentance(sentence_b)
        mlm_tgt_txt = ["[CLS]"] +  self._pad_sequence(sentance_a)[0] + ["[SEP]"] + self._pad_sequence(sentence_b)[0]
        txt_sentance = ["[CLS]"] + prep_a + ["[SEP]"] + prep_b 
        res_itm = torch.tensor([True] + itm_a + [True] + itm_b) # хахахахах в команде на английском хахаха
        res_sentance = torch.tensor([self.word_to_int[word] for word in txt_sentance]) 
        mlm_tgt = torch.tensor([self.word_to_int[word] for word in mlm_tgt_txt])
        #print(print(res_itm.shape), sentance_a, sentence_b)

        return res_sentance, res_itm, nsp_tgt, mlm_tgt


all_dataset = TweetDataset(0.15, sentances_list, 70)
all_loader = DataLoader(dataset=all_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class JointEmbeddings(nn.Module):

    def __init__(self, voacb_size, emb_size):
        super(JointEmbeddings, self).__init__()
        self.emb_size = emb_size

        self.token_embs = nn.Embedding(voacb_size, emb_size)
        self.seg_embs = nn.Embedding(voacb_size, emb_size)

        self.norm_func = nn.LayerNorm(emb_size)

    def _get_pos_embs(self, seq): # we can also use lernable positional embeddings if we add one more nn.Embedding layer
        _batch_size = seq.shape[0]
        seq_len = seq.shape[-1]

        pos = torch.arange(seq_len, dtype=torch.long)
        d = torch.arange(self.emb_size, dtype=torch.long)
        d = (2 * d / self.emb_size)

        pos = pos.unsqueeze(1)
        pos = pos / (10000 ** d)
        pos[:, ::2] = torch.sin(pos[:, ::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        return pos.expand(_batch_size, *pos.shape)
    
    def forward(self, seq):
        seq_len = seq.shape[-1]
        pos_embs = self._get_pos_embs(seq)

        seg_pos = torch.zeros_like(seq)
        seg_pos[:,  seq_len//2 + 1:] = 1
        res_embs = pos_embs + self.token_embs(seq) + self.seg_embs(seg_pos)
        return self.norm_func(res_embs)
    
class AttentionHead(nn.Module):

    def __init__(self, dim_inpt, dim_out): #dim_inpt = emb_size
        super(AttentionHead, self).__init__()
        self.dim_inpt = dim_inpt

        self.query = nn.Linear(dim_inpt, dim_out)
        self.key = nn.Linear(dim_inpt, dim_out)
        self.value =nn.Linear(dim_inpt,dim_out)

    def forward(self, seq, mask): # self attention
        
        query, key, value = self.query(seq), self.key(seq), self.value(seq)

        scale = query.shape[1] ** 0.5
        scores = torch.bmm(query, key.transpose(1,2)) / scale # use bmm not @ because of the stupid batches that I conveniently always forget about, thy actually should end right now
        mask = mask.view(mask.shape[0],1,-1)
        scores = scores.masked_fill_(mask, -1e9)
        attetion = F.softmax(scores, dim=-1)
        context = torch.bmm(attetion, value)
        return context 

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, dim_inpt, dim_out):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([
            AttentionHead(dim_inpt, dim_out) for _ in range(num_heads)
        ])

        self.fc = nn.Linear(dim_out * num_heads, dim_inpt)
        self.norm = nn.LayerNorm(dim_inpt)

    def forward(self, x, mask):
        attentions = [head(x,mask) for head in self.heads]
        scores = torch.cat(attentions, dim=-1)
        scores = self.fc(scores)
        return self.norm(scores)
    
class Encoder(nn.Module):

    def __init__(self, dim_inpt, dim_out, num_heads=4, dropout=0.2): # Mista disliked
        super(Encoder, self).__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads, dim_inpt, dim_out)
        self.ff = nn.Sequential(
            nn.Linear(dim_inpt, dim_out),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_out, dim_inpt),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_inpt)
        
    
    def forward(self, x, mask):
        atten = self.multi_head_attention(x, mask)
        ff = self.ff(atten)
        return self.norm(ff)
    
class BERT(nn.Module):

    def __init__(self, emb_size, dim_out, vocab_size, num_heads=4, dropout=0.2 ):
        super(BERT, self).__init__()
        self.Encoder = Encoder(emb_size, dim_out, num_heads=num_heads, dropout=dropout)
        self.Embeddings = JointEmbeddings(vocab_size, emb_size)

        self.token_pred_layer = nn.Linear(emb_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.classification_layer = nn.Linear(emb_size, 1)
    
    def forward(self, x, mask):
        embeddings =  self.Embeddings(x)
        ecoded = self.Encoder(embeddings, mask)
        
        token_pred = self.token_pred_layer(ecoded)
        #print(x.shape)
        first_word = ecoded[:,0,:]
        return self.softmax(token_pred), self.classification_layer(first_word)
    
    
lr = 0.01
model = BERT(64, 16, vocab_size)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
epochs = 4
nsp_criterion = nn.BCEWithLogitsLoss()
mlm_criterion = nn.NLLLoss() # masked language model






def train_model(num_epochs):
    
    for _ in range(num_epochs):

        for i, batch in enumerate(all_loader):
            optimizer.zero_grad()
            xs, masks, nsp_tgt,mlm_tgt  = batch

            mlm_pred, nsp_pred = model(xs, masks)
            inverse_token_mask = ~masks
            inverse_token_mask = inverse_token_mask.unsqueeze(-1).expand_as(mlm_pred)
            #print(mlm_pred.shape)
            mlm_pred = mlm_pred.masked_fill(inverse_token_mask, 0)

            mlm_loss = mlm_criterion(mlm_pred.transpose(1,2), mlm_tgt)
            nsp_loss = nsp_criterion(nsp_pred.view(nsp_tgt.shape).float(), nsp_tgt.float())

            if i % 100 == 0:
                print(_,mlm_loss, nsp_loss, f"{i}/{len(all_dataset)} {i/len(all_dataset)*100}%")

            optimizer.step()


train_model(epochs)

            



        

