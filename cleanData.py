# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:09:10 2019

@author: MehmetSanisoglu
"""

import pandas as pd
import os
import sys
from langdetect import detect
import re

oldData = pd.read_csv("website_data.csv", nrows = 5000)
newData = pd.DataFrame(columns= ["username","category","website"])

filters = ["godaddy","access denied","accessdenied","please sign", "please enable", "sign into","lasik"]

#ELIMINATE NON ENGLISH SITES
for index, row in oldData.iterrows():
    language = ""
    #MAKING USE OF langdetect'
    try:
        language = (detect(row["website"]) == "en")

    except:    #NOT A VALID CONTENT'
        language = False

    if(language):
        text = row["website"].lower()
        text = re.sub(r'[^A-Za-z]', ' ', text)
        shortword = re.compile(r'\W*\b\w{1,3}\b')

        """
        CHECK FOR THE FILTERS
        IF ANY OF THE UNWANTED ELEMENTS EXIST IN THE WEBSITE CONTENT,
        ELIMINATE THE SITE
        """
        flag = True
        for each in filters:
            if each in text:
                flag = False

        """
        REMOVE THE SHORTWORDS FROM THE SITE CONTENTS AND CHECK
        WHETHER THEY HAVE A MINIMUM LENGTH OF 200 CHARACTERS
        ADD THEM TO THE NEW DATAFRAME IF IT IS THE CASE
        """
        if(flag):
            text = shortword.sub('', text)
            text = ' '.join(text.split())
            if(index!=0):
                previousList = list( newData[:index]["website"] )

                if(previousList.count(text)==0):
                    if(len(text)>200 ):
                        newData = newData.append(pd.Series([row["username"], row["category"], text], index=oldData.columns ), ignore_index=True)
            else:
                if(len(text)>200 ):
                    newData = newData.append(pd.Series([row["username"], row["category"], text], index=oldData.columns ), ignore_index=True)

#ORDER THE WEBSITE CONTENTS BASED ON THEIR CATEGORIES AND OUTPUT THEM ON A CSV FILE'
path = os.path.join(sys.path[0], "output_cleaned.csv")
output = newData.sort_values('category')
output.to_csv(path)
