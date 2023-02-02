# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:19:30 2022

@author: 253364
"""

import os
import pandas as pd, csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import confusion_matrix
import seaborn as sns

import math
import random
import pickle

from functions import get_training_testing, convert_to_numeric, save_agent, load_agent, convert_to_lower




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
            print("Epoch ", epoch)
        print(losses)
        return losses


with open('ix_to_word.pickle', 'rb') as f:
    ix_to_word = pickle.load(f)
   
with open('word_to_ix.pickle', 'rb') as f:
    word_to_ix = pickle.load(f)

model = load_agent("LangModel.pickle")


model.get_logprob(["pitch","collected"],".")

word=model.nextlikely(["hi","there"])


print(word)

path = "C:\\Users\\Student\\Documents\\y1sussexAI\\AdvancedNLP\\CourseWork1\\Dataset\\sentence-completion\\"

features_path = "testing_data.csv"
answers_path = "test_answer.csv"

questions = os.path.join(path, features_path)
answers = os.path.join(path, answers_path)


with open(questions) as instream:
    csvreader = csv.reader(instream)
    lines = list(csvreader)
    
qs_df = pd.DataFrame(lines[1:], columns=lines[0])
qs_df.head()

print(qs_df.head())

with open(answers) as instream:
    csvreader = csv.reader(instream)
    lines = list(csvreader)
    
ans_df = pd.DataFrame(lines[1:], columns=lines[0])
ans_df.head()

print(qs_df.columns)
print(ans_df.columns)




#preprocess the data so the quuestions can be solved


questions = qs_df['question'] #find the question
A = qs_df['a)']                 #split up the different options
B = qs_df['b)']
C = qs_df['c)']
D = qs_df['d)']
E = qs_df['e)']

ans_df['answer'] = ans_df['answer'].apply(convert_to_numeric) #convet everything to numbers from letters

correct_answers = ans_df['answer']

out_list = ['A', 'B', 'C', 'D', 'E']

model_ans = []

print(questions[:5])
print(ans_df['answer'][:5])

for i, question in enumerate(questions):        #loop through all questions
    
    question = question.split()
    
    print(question)

    index = question.index('_____')     #find the index of the gap
    
    wordB = question[index - 1]         #find the words for the bigram
    wordA = question[index - 2]
    
    print(wordA, wordB)
    
    targets = [A[i], B[i], C[i], D[i], E[i]] #find the targets 
    
    probs = []
    
    for target in targets:
        try:
            prob = model.get_logprob([wordA, wordB], target) #loop through the targets and then make prediction
            probs.append(prob)
        except KeyError:
            probs.append(-1)
        
    highest_probs = max(probs)                                  #find the highest probability
    value_index = probs.index(highest_probs)                    #find the index of highest porbability
    final_ans = out_list[value_index]                           #find the final ans
    final_word = targets[value_index]                           #get the final word
    print("Word: {}, Option: {}".format(final_word, final_ans)) 
    model_ans.append(final_ans)                                 #add to model predictions list
    
    



model_ans_df = pd.DataFrame(model_ans, columns=["ans"])             #convert to pandas dataframe


save_ans = model_ans_df["ans"].apply(convert_to_lower)              #convert to lowercase

save_ans.to_csv("WS_results.csv")                   #convert results to CSV
model_ans_df.to_csv("Neural_Results.csv")

model_ans_df["ans"] = model_ans_df["ans"].apply(convert_to_numeric)

print("Correct Answers: {} Model Answers: {}".format(correct_answers[:5], model_ans_df[:5]))

score = 0

lists = zip(correct_answers, model_ans_df['ans'])   #join the lists together

score = 0

for i in lists:                 #loop through to find a score
    if i[0] == i[1]:
        score = score + 1 
    else:
        pass
    
  

accuracy = score/len(correct_answers) #find the accuracy
print("Model Accuracy", accuracy)

conf_mat = confusion_matrix(correct_answers, model_ans_df['ans'])                   #make confussion matrix
sns.heatmap(conf_mat, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)









