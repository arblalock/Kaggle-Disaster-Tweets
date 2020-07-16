# %%
#Import
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pre_process
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers, regularizers
from tensorflow.keras.constraints import max_norm
os.chdir('/workspaces/MachineLearning/Kaggle/Kaggle-Disaster-Tweets/')
%load_ext autoreload
%autoreload 2

# %%
#Data loading
train_data = pd.read_csv('./data/train.csv')
submission_data = pd.read_csv('./data/test.csv')

#Globals
TEST_SIZE = 0.1
RAND = 10
features = ['text']
# features = ['keyword', 'location', 'text']
target = ['target']

# Exploring
# print(train_data.head())
# print(train_data.describe())
# print(train_data.isnull().sum())
# print('Percentage of tweets about disaster: '+ str(train_data['target'].sum()/train_data['target'].count()))
# disaster_text = train_data[train_data['target'] == 1]['text'].values
# non_disaster_text = train_data[train_data['target'] == 0]['text'].values
# wordcloud = WordCloud(
#     width = 3000,
#     height = 2000,
#     background_color = 'black',
#     stopwords = STOPWORDS).generate(str(disaster_text))
# fig = plt.figure(
#     figsize = (40, 30),
#     facecolor = 'k',
#     edgecolor = 'k')
# plt.imshow(wordcloud, interpolation = 'bilinear')
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.title = 'Disaster Word Cloud'
# plt.show()

# wordcloud = WordCloud(
#     width = 3000,
#     height = 2000,
#     background_color = 'black',
#     stopwords = STOPWORDS).generate(str(non_disaster_text))
# fig = plt.figure(
#     figsize = (40, 30),
#     facecolor = 'k',
#     edgecolor = 'k')
# plt.imshow(wordcloud, interpolation = 'bilinear')
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.title = 'Non Disaster Word Cloud'
# plt.show()

#Data processing
train_df = pre_process.create_df(train_data, features, target)
submission_df = pre_process.create_df(submission_data, features, target)
train_raw, test_raw = train_test_split(train_df, test_size=TEST_SIZE, random_state=RAND)
# train = pre_process.create_embedding_df(train_raw, features)
# test = pre_process.create_embedding_df(test_raw, features)
x_train = train_raw[features].values.ravel()
y_train = train_raw[target].values.ravel()
x_test= test_raw[features].values.ravel()
y_test= test_raw[target].values.ravel()


# %%
# Train Model
# Training settings
BATCH_SIZE = 1
EPOCS = 30
LEARNING_RATE = 0.0001
DROPOUT = 0
SHUFFLE = True
EMBED_SIZE = 512
L2 = 1e-3

opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# model
model = tf.keras.Sequential([
    layers.Lambda(pre_process.UniversalEmbedding,
	output_shape=(EMBED_SIZE,))
  layers.Dropout(DROPOUT),
  layers.Dense(128, activation='relu', activity_regularizer=regularizers.l2(L2)),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=opt,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train           
history = model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=EPOCS,
          shuffle=SHUFFLE)

# plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
# y_pred = model.predict(x_test)
y_pred = model.predict_classes(x_test)
con_mat = tf.math.confusion_matrix(labels=y_test.ravel(), predictions=y_pred.ravel()).numpy()
target_names = ['not a disaster', 'disaster']
print(classification_report(y_test, y_pred, target_names=target_names))