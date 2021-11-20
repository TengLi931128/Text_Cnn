#!/usr/bin/env python
# coding: utf-8


"""
This is a CNN which is used in sentiment classification, idea of Kim's CNN.
Teng Li
21.09.2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import gensim
import numpy as np
from config import DefaultConfig

# first of all get config
Conf = DefaultConfig()

# then we can load word2vec
model = gensim.models.KeyedVectors.load_word2vec_format('/home/teng/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz',
                                                        binary=True,limit=20000)
# first add 'unk' and 'pad'to our model
model['unk'] = Conf.unk_vec
model['pad'] = Conf.pad_vec
# the weights of Embedding layer are
W2V_weights = torch.FloatTensor(model.vectors)
# the vocab of this Embedding is
Vocab = model.wv.vocab
# get index of 'unk' and 'pad'
Unk_index = Vocab['unk'].index
Pad_index = Vocab['pad'].index

BATCH_SIZE = Conf.batch_size
EPOCHS = Conf.epochs
WORD_VECTOR_SIZE = 300 # The size of word vector
FILTER_SIZE = Conf.filter_size # 3 different size of filter
FILTER_NUM = Conf.filter_num # The number of each size filter
#DEVICE = Conf.device


class ConvNet(nn.Module):# inherit from nn.Module
    
    def __init__(self):#init the module
        super().__init__()
        #Embedding layer
        self.embed = nn.Embedding.from_pretrained(W2V_weights)
        #Conv layers
        self.conv1 = nn.Conv2d(1,FILTER_NUM[0],(FILTER_SIZE[0],WORD_VECTOR_SIZE))
        self.conv2 = nn.Conv2d(1,FILTER_NUM[1],(FILTER_SIZE[1],WORD_VECTOR_SIZE))
        self.conv3 = nn.Conv2d(1,FILTER_NUM[2],(FILTER_SIZE[2],WORD_VECTOR_SIZE))
        #Fc layer
        self.fc1 = nn.Linear(3*100, 80) #input 20*10*10, out 500
        self.fc2 = nn.Linear(80, 20)
        self.fc3 = nn.Linear(20, 2)
        
        
    def forward(self,x):#forward propagation
        #Embedding layer
        x = self.embed(x)   # batch*1*N(word_num in doc) -> batch*1*N*WORD_VECTOR_SIZE 
        
        #Conv layers
        out1 = self.conv1(x) # batch*1*N*WORD_VECTOR_SIZE -> batch*100*N*WORD_VECTOR_SIZE
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out1 = torch.tanh(out1)
        out2 = torch.tanh(out2)
        out3 = torch.tanh(out3)
        
        #Max over time pooling layer
        max1,_ = torch.max(out1,2)
        max2,_ = torch.max(out2,2)
        max3,_ = torch.max(out3,2)
        
        #Concatenates the features and reshape for FC layer
        Max = torch.cat((max1,max2,max3),1)
        Max = Max.view(BATCH_SIZE,-1)

        #Fc layer
        Out = self.fc1(Max) # batch*300 -> batch*80
        Out = torch.tanh(Out)
        Out = self.fc2(Out) # batch*80 -> batch*20
        Out = torch.tanh(Out)
        Out = self.fc3(Out) # batch*20 -> batch*2
        Out = f.log_softmax(Out,dim=1) #"dim=1" means logsoftmax along 2 sentiment(pos or neg)
        
        return Out
     
    def predict(self,x):
        # this is only used for predict a single instance (same as forward() but Batch_size == 1)
        x = self.embed(x)   
        out1 = self.conv1(x) 
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out1 = torch.tanh(out1)
        out2 = torch.tanh(out2)
        out3 = torch.tanh(out3)
        max1,_ = torch.max(out1,2)
        max2,_ = torch.max(out2,2)
        max3,_ = torch.max(out3,2)
        Max = torch.cat((max1,max2,max3),1)
        Max = Max.view(1,-1)
        Out = self.fc1(Max) # batch*300 -> batch*80
        Out = torch.tanh(Out)
        Out = self.fc2(Out) # batch*80 -> batch*20
        Out = torch.tanh(Out)
        Out = self.fc3(Out) # batch*20 -> batch*2
        Out = f.log_softmax(Out,dim=1) #"dim=1" means logsoftmax along 2 sentiment(pos or neg)
        if Out[0][0]> Out[0][1]:
            y_hat = 'POS'
        else:
            y_hat = 'NEG'
        return y_hat

def word2index(words):
    '''
    doc -> index
    '''
    L = len(words)
    words_id = torch.LongTensor(1,L)
    for l in range(L):
        word = words[l]
        if word in Vocab:
            words_id[0][l] = torch.tensor(Vocab[word].index)
        else:
            words_id[0][l] = torch.tensor(Unk_index)
    
    return words_id


def dynamical_padding(Dataset):
    '''
    padding for each batch
    '''
    len_max = 0
    #get max length
    for d in Dataset:
        l = len(d[0][0])
        if l>len_max:
            len_max = l
    #padding
    texts_id = []
    labels = []
    for d in Dataset:
        words_id = d[0][0][:len_max].tolist()
        label = d[1]
        if label == 'POS':
            label = 0
        else:
            label = 1
        labels.append(label)
        l = len(words_id)
        if l<len_max:
            padding = [Pad_index for _ in range(len_max-l)]
            words_id.extend(padding)
        texts_id.append(words_id)
    #list-> tensor
    labels = torch.tensor(labels)
    texts_id = torch.tensor(texts_id,dtype=torch.long)
    texts_id = texts_id.unsqueeze(1)
    return texts_id,labels

