# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:21:53 2019

@author: MehmetSanisoglu
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
testText = "grange country farm ranch garden grange open  canning supplies fertilizers fruit trees garlic leeks little farmers onions potatoes garden home beneficial insects fertilizer pest control blade sharpening gardening tools equipment home decor gifts lawn pasture care plants seeds soils veggie starts pets cats dogs mobile pigs small animals events contact about your grange delivery membership donations clothing scout uniforms accessories boots western wear work wear english riding wear feed supplies alpacas llamas chickens ducks goats straw swine hogs horse tack supplies wild bird squirrel feed fencing electric fencing standard fencing fuel ethanol free gasoline diesel road tractor diesel propane burn logs burn pellets grange guide horse blanket fitting backyard chickens backyard ducks canning supplies fertilizers fruit trees garlic leeks little farmers onions potatoes garden home beneficial insects fertilizer pest control blade sharpening gardening tools equipment home decor gifts lawn pasture care plants seeds soils veggie starts pets cats dogs mobile pigs small animals events contact local since long before that thing been serving neighbors friends farmers since your only local ethanol free station like energy drink engine farmer lettuce help grow stuff howdy come fashioned hospitality know horse sense farmer matter what work pleasure part small some large others connected farming that part that kicks when squeeze lemon thump watermelon smell flowers that part grange wants celebrate equip full service farm lawn garden store with very good looking uber knowledgeable employees help farm your back your backyard everything between wait there more full service supply store work western clothing store hardware store equine tack supply store full service diesel ethanol free fuel stop there more breyer horses canning supplies truth just gotta come coffee chick schedule events mobile clinic good neighbor breyer pressure canning class homesteading where live follow facebook more details about each event take look around ipad users click here take look around meet team john general manager michelle operations belma marketing rachel inventory jeanmarie merchandising michael lawn garden nate delivery manager vania front warehouse maureen financial manager rants raves inquiries full name email message commentsthis field validation purposes should left unchanged this iframe contains logic required handle ajax powered gravity forms opening hours monday friday saturday sunday gilman blvd issaquah washington follow copyright grange rights reserved website webworks first hear about sales events latest news annoying email namethis field validation purposes should left unchanged this iframe contains logic required handle ajax powered gravity forms"

path = os.path.join(sys.path[0], "output_combined.csv")
df = pd.read_csv(path)

df = df[['category', 'website']]
df = df[pd.notnull(df['website'])]
df.columns = ['category', 'website']

df['category_id'] = df['category'].factorize()[0]
category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

# GET THE TERM FREQUENCY AND INVERSE DOCUEMNT FREQUENCY scores of the words all at once
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.website).toarray()
labels = df.category_id

#TEST CASE WITH A TEST CORPUS
X_train, X_test, y_train, y_test = train_test_split(df['website'], df['category'], random_state = 0)
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = LinearSVC().fit(X_train_tfidf, y_train)

print(clf.predict(count_vect.transform([testText])))

# USING THE TEST DATA
model = LinearSVC()
X_train, X_test, y_train, y_test = train_test_split(features, labels, df.index, test_size=0.6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



# FIND THE MOST USED UNIGRAMS AND BIGRAMGS
singles = ""
doubles = ""

model.fit(features, labels)
N = 2
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

# CREATE A WORDCLOUD TO DISPLAY THE WORDS
vectorizer = CountVectorizer(ngram_range=(2,2))
testData = vectorizer.fit_transform(bigrams)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords= STOPWORDS,
                          max_words=100,
                          max_font_size=100,
                          random_state=42,
                          width=1600,
                          height=800).generate(singles)
print(wordcloud)
fig = plt.figure(1)
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

# PRINT THE ACCURACY SCORE
from sklearn import metrics

print(metrics.classification_report(y_test, y_pred, target_names=df['category'].unique()))
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
