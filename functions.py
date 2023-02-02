# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:35:43 2022

@author: 253364
"""

import os
import random#
import torch as T
import matplotlib.pyplot as plt 
import numpy as np

def get_training_testing(training_dir, split=0.5):
    filenames=os.listdir(training_dir)
    n=len(filenames)
    print("There are {} files in the training directory: {}".format(n,training_dir))
    #random.seed(53) #if you want the same random split every time
    random.shuffle(filenames)
    index=int(n*split)
    return(filenames[:index],filenames[index:])

def convert_to_numeric(x):
    
    if x == "a" or x == "A":
        return 0
    elif x == "b" or x == "B":
        return 1
    elif x == "c" or x == "C":
        return 2
    elif x == "d" or x == "D":
        return 3
    else:
        return 4
    
def convert_to_lower(x):
    if x == "a" or x == "A":
        return 'a'
    elif x == "b" or x == "B":
        return 'b'
    elif x == "c" or x == "C":
        return 'c'
    elif x == "d" or x == "D":
        return 'd'
    else:
        return 'e'

def save_agent(state, filename):
    T.save(state, filename)
    
    
def load_agent(file_name):
    
    print("Loading Check point")
    
    model = T.load(file_name)
 
    return model
        

def get_left_context(sent_tokens,window,target="_____"):
    found=-1
    for i,token in enumerate(sent_tokens):
        if token==target:
            found=i
            break 
            
    if found>-1:
        return sent_tokens[i-window:i]
    else:
        return []