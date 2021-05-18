# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:21:53 2019

@author: MehmetSanisoglu
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from wordcloud import WordCloud, STOPWORDS


path = os.path.join(sys.path[0], "output_combined.csv")
df = pd.read_csv(path)
df = df[['category', 'website']]
df = df[pd.notnull(df['website'])]
df.columns = ['category', 'website']

df['category_id'] = df['category'].factorize()[0]
category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

# Get tf-idf scores of the words all at once
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.website).toarray()
labels = df.category_id

# Model
model = LinearSVC()
X_train, X_test, y_train, y_test = train_test_split(features, labels, df.index, test_size=0.6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Find the most common unigrams and bigrams
singles = ""
doubles = ""

model.fit(features, labels)
N = 5
for category, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]

  for i in range(2):
      singles = singles + " " + unigrams[i]
      doubles =  doubles + " " + bigrams[i]
  print("# '{}':".format(category))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

# Create a wordcloud to display the words
vectorizer = CountVectorizer(ngram_range=(2,2))
testData = vectorizer.fit_transform(bigrams)

wordcloud = WordCloud(
  background_color='white',
  stopwords=STOPWORDS,
  max_words=100,
  max_font_size=100,
  random_state=42,
  width=1600,
  height=800
).generate(singles)

fig = plt.figure(1)
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

# Print the accuracy
print(metrics.classification_report(y_test, y_pred, target_names=df['category'].unique()))
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
