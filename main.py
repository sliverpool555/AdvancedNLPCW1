# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:51:35 2022

@author: 253364
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from nltk import word_tokenize as tokenize
import math
import random



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
    
    def train(self,inputngrams,loss_function=nn.NLLLoss(),lr=0.001,epochs=10):
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
        print(losses)



torch.manual_seed(1)

word_to_ix = {"hello": 0, "world": 1}   #create index for each word

embeds = nn.Embedding(2, 5)             #2 words, 5 dimental embeddings

lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long) #identify the index into this embedding matrix for the word of interest - this is stored in a 1-d tensor


hello_embed = embeds(lookup_tensor)     #find the embedding of interest
print(hello_embed)

current_tensor = torch.tensor([word_to_ix["world"]], dtype =torch.long)
print(embeds(current_tensor))



##N-Gram Language Modelling

CONTEXT_SIZE = 2  #this is the amount of preceding context to consider
EMBEDDING_DIM = 10  #this is the dimension of the embeddings


# We will use Shakespeare Sonnet 2
test_sentence = ["__END","__START"]+tokenize("""When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""")+["__END"]


# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the last 3, just so you can see what they look like
#print(trigrams[-3:])


#find the vocabulary and create the index
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}
print(word_to_ix)


#Now the model
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
model.train(trigrams, word_to_ix)

model.get_logprob(["his","field"],".", word_to_ix)

word=model.nextlikely(["his","field"], word_to_ix,ix_to_word)
print(word)


model.generate()









