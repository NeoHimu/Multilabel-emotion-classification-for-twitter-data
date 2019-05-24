import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import pandas as pd
import re
import seaborn as sns
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy import linalg as LA


module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)


def similarity(u, v):
    #return np.dot(u, v)
    return (1-np.arccos(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)))/3.1416)    

df = pd.read_csv("train - train.csv")

sentences = []

for sent in df["Tweet"]:
	temp_sent = []
	for word in sent.split():
		if(word.startswith("@")):
			continue
		elif(word.startswith("#")):
			temp_sent.append(word[1:])
		else:
			temp_sent.append(word)
			
	sentences.append(" ".join(temp_sent))


#print(sentences)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#sentence_embedding = np.array([0])
with tf.Session(config=config) as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  sentence_embedding = np.array(session.run(embed(sentences)))
  print(sentence_embedding.shape)  
  df['embedding'] = pd.Series(sentence_embedding.tolist()) 
  
#print(df['embedding'])

emotions = []

emotion_names = ['anger','anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust']

for emotion_name in emotion_names:
  temp = [0.0]*512
  for vec in df['embedding'][df[emotion_name]==1].values:	
    for idx, ele in enumerate(vec):
      temp[idx] += ele

  #print(np.asarray(temp, dtype='float32')/(1.0*len(df['embedding'][df['anger']==1].values)))
  emotions.append(np.asarray(temp, dtype='float32')/(1.0*len(df['embedding'][df[emotion_name]==1].values)))
print(emotions)
pickle.dump(emotions, open("emotion_embeddings.pickle", "wb"))
