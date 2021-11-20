#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Text_Cnn import word2index

#clean datasets for Word2vec Embedding
def clean_datasets(docs):
    """
    clean data for Word2vec Emnbedding
    """
    punct = [',','.',':','(',')','?','!','-']
    preposition = ['to','of','and','a']
    remove_list = punct + preposition
    for docid in docs:
        doc = docs[docid]
        #remove words
        doc = list(filter(lambda x: x not in remove_list, doc))
        #replace words
        for i,word in enumerate(doc):
            if word == "'s":
                doc[i] = 'is'
            if word == "n't":
                doc[i] = 'not'
        #return cleaned doc    
        docs[docid] = doc
    return docs

# raw data -> dataset
# some necessary function
def get_label(dataset):
    return dataset.classification
def get_docid(dataset):
    return c.annotation_id

# create the dataset class
class Movie_Classif_Dataset(Dataset):
    """
    create the dataset class for Movie sentiment classification
    """
    def __init__(self, docs, annos):
        # run once when init
        self.docs = docs
        self.annos = annos
        self.labels = list(map(get_label,annos))
        
    def __len__(self):
        #returns the number of samples in our dataset
        return len(self.labels)
    
    def __getitem__(self, idx):
        #returns a sample from the dataset at the given index
        label = self.labels[idx]
        docid = self.annos[idx].annotation_id
        text = self.docs[docid]
        words_id = word2index(text)
        #sample = {"Text": text, "Class": label}
        return words_id,label
