# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:05:40 2019

@author: MehmetSanisoglu
A python script to traverse the provided websites and

"""
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import collections
import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder
from numpy import array

from sklearn.feature_extraction.text import TfidfVectorizer

path = os.path.join(sys.path[0], "output_cleaned.csv")
df = pd.read_csv(path)
size = df.shape[0]
allwords = []
cleanedFrame = pd.DataFrame(columns= ["category","content"])

inputText = ""
tempTopic = df.category[0]
contentsFrame = pd.DataFrame(columns= ["category","content"])

for index, row in df.iterrows():
    currentTopic = row["category"]
    if(currentTopic == tempTopic):
        if(index+1 == size):    #END OF LAST CATEGORY

            inputText += row["website"] + " "
            contentsFrame = contentsFrame.append(pd.Series([row["category"], inputText], index=contentsFrame.columns ), ignore_index=True)

        else:   #APPEND THE WEBSITE CONTENTS
            inputText += row["website"] + " "

    else:   #CHANGE OF CATEGORY

            contentsFrame = contentsFrame.append(pd.Series([tempTopic, inputText], index=contentsFrame.columns ), ignore_index=True)
            tempTopic = currentTopic
            inputText = row["website"] + " "


keyCounts = pd.DataFrame(columns= ["category","key"])
keyCounts["category"] = contentsFrame["category"].values

indexer = 0


# LEMMATIZE AND TOKENIZE EACH CORPUS IN THE DATAFRAME

for each in contentsFrame["category"]:

    lemmatizer = WordNetLemmatizer()
    word_list = nltk.word_tokenize(each)

    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    mylist = nltk.word_tokenize(lemmatized_output)

    removeWords = ["facebook","twitter","pinterest","thisblogthis","http","link"]
    newList = []
    maxIndex = len(mylist)


    # ELIMINATE CONTENTS RELATED TO CERTAIN SITES AND DOMAINS

    for index in range(maxIndex):
        flag = False

        for each in removeWords:
            if(each in mylist[index]):
                flag = True

        if(not flag):
            newList.append(mylist[index])

    stopWords = set(stopwords.words("english"))
    wordsFiltered = []
    for w in newList:
        if w not in stopWords:
            wordsFiltered.append(w)

    #SELECT THE MOST COMMON WORDS
    mostCommon = collections.Counter(wordsFiltered).most_common(10)

    trace = 0
    commons = []
    for each in mostCommon:
        commons.append(each[0])


    keyCounts["key"][indexer] = commons
    indexer+=1
    for each in commons:
        allwords.append(each)

    cleanedFrame["category"] = each
    cleanedFrame["content"] = allwords

#ORDER THE WEBSITE CONTENTS BASED ON THEIR CATEGORIES AND OUTPUT THEM ON A CSV FILE'
path = os.path.join(sys.path[0], "output_combined.csv")
output = cleanedFrame.sort_values('category')
output.to_csv(path)
