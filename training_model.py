# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:41:37 2022

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

with open('ix_to_word.pickle', 'rb') as f:
    ix_to_word = pickle.load(f)
   
with open('word_to_ix.pickle', 'rb') as f:
    word_to_ix = pickle.load(f)
    
with open('vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)
    
with open('trigrams.pickle', 'rb') as f:
    trigrams = pickle.load(f)


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def get_logprob(self,context,target):
        #return the logprob of the target word given the context
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        log_probs = self.forward(context_idxs)
        target_idx=torch.tensor(word_to_ix[target],dtype=torch.long)
        return log_probs.index_select(1,target_idx).item()
        
        
    def nextlikely(self,context):
        #sample the distribution of target words given the context
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        log_probs = self.forward(context_idxs)
        probs=[math.exp(x) for x in log_probs.flatten().tolist()]
        t=random.choices(list(range(len(probs))),weights=probs,k=1)
        return ix_to_word[t[0]]
    
    def generate(self,limit=20):
        #generate a sequence of tokens according to the model
        tokens=["__END","__START"]
        while tokens[-1]!="__END" and len(tokens)<limit:
            current=self.nextlikely(tokens[-2:])
            tokens.append(current)
        return " ".join(tokens[2:-1])
    
    def train(self,inputngrams,loss_function=nn.NLLLoss(),lr=0.001,epochs=1):
        optimizer=optim.SGD(self.parameters(),lr=lr)
        
        losses=[]
        for epoch in range(epochs):
            total_loss = 0
            for context, target in inputngrams:

                # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                # into integer indices and wrap them in tensors)
                context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
                self.zero_grad()

                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                log_probs = self.forward(context_idxs)

                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a tensor)
                loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()
            
                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()
            losses.append(total_loss)
            print("Epoch ", epoch)
        print(losses)
        return losses
      


    


print("Model is started")
CONTEXT_SIZE = 2  #this is the amount of preceding context to consider
EMBEDDING_DIM = 10  #this is the dimension of the embeddings

model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
print("Train model")
losses = model.train(trigrams)
print("Finished Trainning")

plt.plot(losses)
plt.show()

save_agent(model, "LangModel.pickle")

model.get_logprob(["pitch","collected"],".")

word=model.nextlikely(["pitch","collected"])
print(word)

model.generate()



