# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:05:40 2019

@author: MehmetSanisoglu
A python script to traverse the provided websites and

"""
import collections
import pandas as pd
import os
import sys

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

path = os.path.join(sys.path[0], "output_cleaned.csv")
df = pd.read_csv(path)
size = df.shape[0]
allwords = []
cleaned_frame = pd.DataFrame(columns= ["category","content"])

inputText = ""
tempTopic = df.category[0]
contentsFrame = pd.DataFrame(columns= ["category","content"])

for index, row in df.iterrows():
    currentTopic = row["category"]
    # Same category
    if currentTopic == tempTopic:
        # End of a category
        if index+1 == size:    
            inputText += row["website"] + " "
            contentsFrame = contentsFrame.append(pd.Series([row["category"], inputText], index=contentsFrame.columns ), ignore_index=True)
        # More rows to iterate in the same category
        else:
            inputText += row["website"] + " "
    # Change of category
    else:   
            contentsFrame = contentsFrame.append(pd.Series([tempTopic, inputText], index=contentsFrame.columns ), ignore_index=True)
            tempTopic = currentTopic
            inputText = row["website"] + " "

key_counts = pd.DataFrame(columns= ["category","key"])
key_counts["category"] = contentsFrame["category"].values

indexer = 0

# Lemmatize and tokenize each content in the dataframe
for each in contentsFrame["category"]:

    lemmatizer = WordNetLemmatizer()
    word_list = nltk.word_tokenize(each)

    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    mylist = nltk.word_tokenize(lemmatized_output)

    filter_words = ["facebook","twitter","pinterest","thisblogthis","http","link"]
    filtered_content = []
    maxIndex = len(mylist)

    # ELIMINATE CONTENTS RELATED TO CERTAIN SITES AND DOMAINS
    for index in range(maxIndex):
        flag = False
        for each in filter_words:
            if each in mylist[index]:
                flag = True
        # Safe to add to the aggregate
        if not flag:
            filtered_content.append(mylist[index])

    stop_words = set(stopwords.words("english"))
    words_filtered = []
    for w in filtered_content:
        if w not in stop_words:
            words_filtered.append(w)

    # Get 10 most common words
    most_common = collections.Counter(words_filtered).most_common(10)

    commons = [each[0] for each in most_common]

    key_counts["key"][indexer] = commons
    indexer+=1
    for each in commons:
        allwords.append(each)

    cleaned_frame["category"] = each
    cleaned_frame["content"] = allwords

# Order the website contents based on their categories
path = os.path.join(sys.path[0], "output_combined.csv")
output = cleaned_frame.sort_values('category')
output.to_csv(path)
