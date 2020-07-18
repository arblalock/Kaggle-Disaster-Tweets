from tqdm import tqdm
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
#Load Embeddings
embed = None
if embed == None:
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def create_df(data, features, target):
    data = data.fillna('missing')
    return data.filter(features+target)


def universal_embedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)))
