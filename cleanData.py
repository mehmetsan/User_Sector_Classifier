# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:09:10 2019

@author: MehmetSanisoglu
"""

import os
import sys
import re
import pandas as pd

from language import apply_filters, check_english

oldData = pd.read_csv("website_data.csv", nrows = 5000)
newData = pd.DataFrame(columns= ["username","category","website"])

# Eliminate non-english sites
for index, row in oldData.iterrows():
    # Making use of langdetect
    language = check_english( row["website"] )

    if language:
        text = row["website"].lower()
        text = re.sub(r'[^A-Za-z]', ' ', text)
        shortword = re.compile(r'\W*\b\w{1,3}\b')

        # Apply filters on the site content
        flag = apply_filters(text)

        # Remove shortwords from the site contents and check minimum
        # 200 characters threshold
        if flag:
            text = shortword.sub('', text)
            text = ' '.join(text.split())
            if index!=0 :
                previousList = list( newData[:index]["website"] )
                if previousList.count(text)==0:
                    if len(text)>200:
                        newData = newData.append(pd.Series([row["username"], row["category"], text], index=oldData.columns ), ignore_index=True)
            else:
                if len(text)>200:
                    newData = newData.append(pd.Series([row["username"], row["category"], text], index=oldData.columns ), ignore_index=True)

# Order website contents based on their categories
path = os.path.join(sys.path[0], "output_cleaned.csv")
output = newData.sort_values('category')
output.to_csv(path)
