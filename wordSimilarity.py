# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:16:17 2022

@author: 253364
"""



#Word Simalirty Code

import pandas as pd, csv
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random

from gensim.models import KeyedVectors

from functions import get_training_testing, convert_to_numeric, save_agent, load_agent, convert_to_lower


filename = "GoogleNewsvectorsnegative300.bin"
mymodel = KeyedVectors.load_word2vec_format(filename, binary=True) #access the language model

"""
print(mymodel.similarity('men', 'women'))


print(mymodel.most_similar(positive=['China','London'],negative=['England']))
print()
print(mymodel.most_similar(positive=['China','London'],negative=['Beijing']))

print()
print(mymodel.most_similar(positive=['China','London','Hong_Kong']))
"""

path = "C:\\Users\\Student\\Documents\\y1sussexAI\\AdvancedNLP\\CourseWork1\\Dataset\\sentence-completion\\"

features_path = "testing_data.csv"  #get questions and answers
answers_path = "test_answer.csv"

questions = os.path.join(path, features_path)
answers = os.path.join(path, answers_path)


with open(questions) as instream:
    csvreader = csv.reader(instream)    #read the data and organise
    lines = list(csvreader)
    
qs_df = pd.DataFrame(lines[1:], columns=lines[0]) #make dataset
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


questions = qs_df['question']   #get the questions in a list
A = qs_df['a)']                 #Get all the potental answers
B = qs_df['b)']
C = qs_df['c)']
D = qs_df['d)']
E = qs_df['e)']

ans_df['answer'] = ans_df['answer'].apply(convert_to_numeric) #convert answers to numentic

correct_answers = ans_df['answer']

out_list = ['A', 'B', 'C', 'D', 'E']

model_ans = []

print(questions[:5])
print(ans_df['answer'][:5])

for i, question in enumerate(questions):
    
    question = question.split() #tokenise the question (I now after writing report have found this issue that could be effecting accuracy :( )
    
    print(question)

    index = question.index('_____') #get index of gap
    
    wordA = question[index - 1]     #get words before gap
    wordB = question[index - 2]
    wordC = question[index - 3]
    wordD = question[index - 4]
    
    print(wordA, wordB)
    
    targets = [A[i], B[i], C[i], D[i], E[i]] #get the target words
    
    probs = []
    
    for target in targets:
        try:
            #prob = mymodel.most_similar(positive=[wordA,wordB],negative=[target])
            prob = mymodel.most_similar(positive=[wordB, wordA, target])       #find the probability of the words using model
            probs.append(prob)
        except:
            prob = random.choice((target, 0))
            probs.append(prob)

    print(probs)
    try:
        highest_probs = max(probs)
    except:
        print("Error")
        highest_probs = probs[0] 
    value_index = probs.index(highest_probs) #find the highest probability
    final_ans = out_list[value_index]           #get final output
    final_word = targets[value_index]           #find the final word and add to answers
    print("Word: {}, Option: {}".format(final_word, final_ans))
    model_ans.append(final_ans)
    
    



model_ans_df = pd.DataFrame(model_ans, columns=["ans"]) #put answers into the dataframe

save_ans = model_ans_df
save_ans = save_ans["ans"].apply(convert_to_lower)      #convert to lowercase

save_ans.to_csv("WS_results.csv")                       #convert to csv

model_ans_df["ans_nums"] = model_ans_df["ans"].apply(convert_to_lower) #convert to lowercase

print("Correct Answers: {} Model Answers: {}".format(correct_answers[:5], model_ans_df[:5]))
                                                                 
                                                                 
path = "C:\\Users\\Student\\Documents\\y1sussexAI\\AdvancedNLP\\CourseWork1\\Dataset\\sentence-completion\\"

answers_path = "test_answer.csv"

answers = os.path.join(path, answers_path) #get the answrs and put into dataframe

with open(answers) as instream:
    csvreader = csv.reader(instream)
    lines = list(csvreader)
    
ans_df = pd.DataFrame(lines[1:], columns=lines[0]) 

correct_answers = ans_df['answer']

conf_mat = confusion_matrix(correct_answers, model_ans_df["ans_nums"])              #create the confussion matrix
sns.heatmap(conf_mat, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)

lists = zip(correct_answers, model_ans_df["ans_nums"]) #join both lists and find score

score = 0

for i in lists:
    print(i)
    if i[0] == i[1]:
        score = score + 1
    else:
        pass
    
  

accuracy = score/len(correct_answers) #find average and print
print("Model Accuracy", accuracy)

