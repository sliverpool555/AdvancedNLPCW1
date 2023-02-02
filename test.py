# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:14:16 2022

@author: 253364
"""

#testing code

sentence = "Hello how are you today would you like ____"

words = sentence.split()

print(words)

index = words.index("____")

wordB = words[index - 1]
wordA = words[index - 2]

print(wordA, wordB)


        