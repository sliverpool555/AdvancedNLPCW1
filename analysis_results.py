# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 00:37:20 2022

@author: 253364
"""


import pandas as pd, csv
import numpy as np

import nltk
from nltk import word_tokenize as tokenize

nltk.download('punkt')


#Analysis program

addr = "Results_for_all.csv"

with open(addr) as instream:
    csvreader = csv.reader(instream)
    lines = list(csvreader)
    
df = pd.DataFrame(lines[1:], columns=lines[0])
df.head()

unigram = df['unigram_pred']
bigram = df['bigram_pred']
word_sim = df['word_sim_pred']
neural_pred = df['neural_pred']
ans = df['Answer']

questions = df['question']


#find the indexs when all models get it correct

score = 0
indexs = []

four_cor = 0
three_cor = 0
two_cor = 0
one_cor = 0
none_cor = 0


for i in range(len(df)):
    
    score = 0
    
    if unigram[i] == ans[i]:
        score = score + 1
    if bigram[i] == ans[i]:
        score = score + 1
    if word_sim[i] == ans[i]:
        score = score + 1
    if neural_pred[i] == ans[i]:
        score = score + 1
        
    if score == 4:
        four_cor = four_cor + 1
        
    if score == 3:
        three_cor = three_cor + 1
        
    if score == 2:
        two_cor = two_cor + 1
        
    if score == 1:
        one_cor = one_cor + 1
        
    if score == 0:
        indexs.append(i)
        none_cor = none_cor + 1
        
print("All correct index's", indexs)
print("Amount in bracket: ", len(indexs))

#Now find the average ___ where all of them get the correct answer

counts = []
lenghts = []
        
for i in indexs:
    count = 0
    question = tokenize(questions[i])
    
    for j in question:
        
        count = count + 1
        
        if j == "_____":
            counts.append(count)
            
       
average = sum(counts) / len(counts)

mean = np.mean(counts)
mode = print(max(set(counts), key = counts.count)) 
medium = np.median(counts)
print("Average gap", average)

print("Mean {}, mode {}, medium {}".format(mean, mode, medium))
        
print("Gaps index is ", counts)





print("Average index of where the gap is when all models get it correct", average)


print("4 Correct) {}, | 3 Correct) {}, | 2 Correct) {}, | 1 Correct 0) {}, | 0 Correct) {}".format(four_cor, three_cor, two_cor, one_cor, none_cor))

print()

print(indexs)

print()

print("Questions all the models got right")
        

for i in indexs:
    print()
    print(i, questions[i], ans[i])
    
    


