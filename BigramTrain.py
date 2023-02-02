# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:23:05 2022

@author: 253364
"""

import os
import pandas as pd, csv
from nltk import word_tokenize as tokenize
import nltk
import numpy as np

from functions import get_training_testing, get_left_context, convert_to_numeric
from language_model import language_model


nltk.download('punkt')


dataset_addr = "C:\\Users\\Student\\Documents\\y1sussexAI\\AdvancedNLP\\CourseWork1\\Dataset\\sentence-completion\\Holmes_Training_Data\\"

train, test = get_training_testing(dataset_addr)



#Trigram model


path = "C:\\Users\\Student\\Documents\\y1sussexAI\\AdvancedNLP\\CourseWork1\\Dataset\\sentence-completion\\"

features_path = "testing_data.csv"
answers_path = "test_answer.csv"

questions = os.path.join(path, features_path) #get the questions and answers
answers = os.path.join(path, answers_path)


with open(questions) as instream:
    csvreader = csv.reader(instream) #get data from the csv file
    lines = list(csvreader)
    
qs_df = pd.DataFrame(lines[1:], columns=lines[0]) #convert to dataframe
qs_df.head()

print(qs_df.head())

with open(answers) as instream:
    csvreader = csv.reader(instream)
    lines = list(csvreader)
    
ans_df = pd.DataFrame(lines[1:], columns=lines[0])
ans_df.head()

print(ans_df.head())


#Preprocessing
print("Preprocessing")
tokens = [tokenize(i) for i in qs_df['question']]                               #tokenise the questions

qs_df['tokens'] = qs_df['question'].map(tokenize)

qs_df['left_context']= qs_df['tokens'].map(lambda x: get_left_context(x,2))     #get the words left of the context

print(qs_df.head())

#Spilt the data into preprocessing

print("Split data")

print(ans_df['answer'])



#Now the tokinizer

qs_df['tokens'] = qs_df['question'].map(tokenize)
qs_df['left_context']=qs_df['tokens'].map(lambda x: get_left_context(x,2))
qs_df.head()


trainingdir = "C:\\Users\\Student\\Documents\\y1sussexAI\\AdvancedNLP\\CourseWork1\\Dataset\\sentence-completion\\Holmes_Training_Data\\"
training,testing=get_training_testing(trainingdir)
MAX_FILES=10   #use a small number here whilst developing your solutions
mylm=language_model(trainingdir=trainingdir,files=training[:MAX_FILES])

#Now the model

class question:
    
    def __init__(self,aline):
        self.fields=aline
    
    def get_field(self,field):
        return self.fields[question.colnames[field]]
    
    def add_answer(self,fields):
        self.answer=fields[1]
   
    def chooseA(self):
        return("a")
    
    def chooserandom(self):
        choices=["a","b","c","d","e"]
        return np.random.choice(choices)
    
    def get_tokens(self):
        return ["__START"]+tokenize(self.fields[question.colnames["question"]])+["__END"]
    
    def choose(self,lm,method="bigram",choices=[]):
        if choices==[]:
            choices=["a","b","c","d","e"]
        context=self.get_left_context(window=1)
        probs=[lm.get_prob(self.get_field(ch+")"),context,methodparams={"method":method}) for ch in choices]
        maxprob=max(probs)
        bestchoices=[ch for ch,prob in zip(choices,probs) if prob == maxprob]
        #if len(bestchoices)>1:
        #    print("Randomly choosing from {}".format(len(bestchoices)))
        return np.random.choice(bestchoices)
    
    def get_left_context(self,window=1,target="_____"):
        found=-1
        sent_tokens=self.get_tokens()
        for i,token in enumerate(sent_tokens):
            if token==target:
                found=i
                break  
            
        if found>-1:
            return sent_tokens[i-window:i]
        else:
            return []
    
    def chooseunigram(self,lm):
        choices=["a","b","c","d","e"]      
        probs=[lm.unigram.get(self.get_field(ch+")"),0) for ch in choices]
        maxprob=max(probs)
        bestchoices=[ch for ch,prob in zip(choices,probs) if prob == maxprob]
        #if len(bestchoices)>1:
        #    print("Randomly choosing from {}".format(len(bestchoices)))
        return np.random.choice(bestchoices)
    
    def predict(self,method="chooseA",model=mylm):
        if method=="chooseA":
            return self.chooseA()
        elif method=="random":
            return self.chooserandom()
        else:
            return self.choose(mylm,method=method)
        
    def predict_and_score(self,method="chooseA"):
        
        #compare prediction according to method with the correct answer
        #return 1 or 0 accordingly
        prediction=self.predict(method=method)
        if prediction ==self.answer:
            return 1
        else:
            return 0
        

class scc_reader:
    
    def __init__(self,qs=questions,ans=answers):
        self.qs=qs
        self.ans=ans
        self.read_files()
        
    def read_files(self):
        
        #read in the question file
        with open(self.qs) as instream:
            csvreader=csv.reader(instream)
            qlines=list(csvreader)
        
        #store the column names as a reverse index so they can be used to reference parts of the question
        question.colnames={item:i for i,item in enumerate(qlines[0])}
        
        #create a question instance for each line of the file (other than heading line)
        self.questions=[question(qline) for qline in qlines[1:]]
        
        #read in the answer file
        with open(self.ans) as instream:
            csvreader=csv.reader(instream)
            alines=list(csvreader)
            
        #add answers to questions so predictions can be checked    
        for q,aline in zip(self.questions,alines[1:]):
            q.add_answer(aline)
        
    def get_field(self,field):
        return [q.get_field(field) for q in self.questions] 
    
    def predict(self,method="chooseA"):
        return [q.predict(method=method) for q in self.questions]
    
    def predict_and_score(self,method="chooseA"):
        scores=[q.predict_and_score(method=method) for q in self.questions]
        return sum(scores)/len(scores)
    
    


SCC=scc_reader()
print(SCC.predict_and_score(method="bigram"))

qs_df["bigram_pred"]=SCC.predict(method="bigram")
print(qs_df.head())

qs_df["unigram_pred"]=SCC.predict(method="unigram")
print(qs_df)




print(mylm.bigram["are"]["matched"])














