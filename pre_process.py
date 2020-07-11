from tqdm import tqdm
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
#Load Embeddings
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def create_df(data, features, target):
    return data.filter(features+target)

def create_embedding_df(dataframe, text_col):
    emb_array = []
    result_df = dataframe.copy()
    for r in tqdm(dataframe[text_col]):
        embedding = embed([r])
        text_emb = tf.reshape(embedding, [-1]).numpy()
        emb_array.append(text_emb)
    
    result_df[text_col] = np.array(emb_array)
    return result_df
