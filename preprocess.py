# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:31:52 2022

@author: 253364
"""


import os
import matplotlib.pyplot as plt
from nltk import word_tokenize as tokenize
from nltk.corpus import stopwords
import nltk

import random
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

from functions import get_training_testing, save_agent, load_agent

nltk.download('punkt')
nltk.download('stopwords')

#Preprocessing the data for model

dataset_addr = "C:\\Users\\Student\\Documents\\y1sussexAI\\AdvancedNLP\\CourseWork1\\Dataset\\sentence-completion\\Holmes_Training_Data\\"

files = os.listdir(dataset_addr)

trainingfiles, heldoutfiles = get_training_testing(dataset_addr)

print(trainingfiles)

stop_words = set(stopwords.words("english"))

#Preprocess the data
datas = []

for doc in trainingfiles:
    
    addr = os.path.join(dataset_addr, doc)
    print(addr)
    
    with open(addr, "r") as file:
        data = file.read()
    
    print(data)
    data = data.lower()
    data = tokenize(data)
    #data = [w for w in data if not w.lower() in stop_words]
    datas = datas + data
    
vocab = set(datas)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}
print(word_to_ix)

print("Write up: ", ix_to_word)

trigrams = [([data[i], data[i + 1]], data[i + 2])
        for i in range(len(data) - 2)]

print(trigrams[-3:])

with open('word_to_ix.pickle', 'wb') as file:
    pickle.dump(word_to_ix, file)
    
with open('ix_to_word.pickle', 'wb') as file:
    pickle.dump(ix_to_word, file)
    
with open('vocab.pickle', 'wb') as file:
    pickle.dump(vocab, file)
    
with open('trigrams.pickle', 'wb') as file:
    pickle.dump(trigrams, file)






    
    





