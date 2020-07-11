import tensorflow_hub as hub
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
os.chdir('/workspaces/MachineLearning/Kaggle/Kaggle-Disaster-Tweets/')
c = pd.read_csv('./data/train.csv')
submission_data = pd.read_csv('./data/test.csv')

#Load Embeddings
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Exploring
print(train_data.head())
print(train_data.describe())
print(train_data.isnull().sum())
print('Percentage of tweets about disaster: '+ str(train_data['target'].sum()/train_data['target'].count()))
text = train_data.text.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
