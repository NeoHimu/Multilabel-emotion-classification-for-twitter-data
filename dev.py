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
from sklearn.cluster import KMeans

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)


def similarity(u, v):
    #return np.dot(u, v)
    return (1-np.arccos(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)))/3.1416)    

df = pd.read_csv("dev - dev.csv")

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

emotion_embeddings = pickle.load(open("emotion_embeddings.pickle", "rb"))

total_count = 0
break_count = 0
for row_num, sentence_dev in enumerate(df['embedding']):
  temp_dot = []
  for emotion_embedding in emotion_embeddings:
    temp_dot.append(np.dot(sentence_dev, emotion_embedding))
  temp_dot = np.asarray(temp_dot)
  #temp_dot = [(ele, idx) for idx, ele in enumerate(temp_dot)]
  #sorted(temp_dot)
  #temp_dot = temp_dot[::-1]
  #temp_pred = [0]*11
  #temp_pred[temp_dot[0][1]]=1
  #temp_pred[temp_dot[1][1]]=1
  #temp_pred[temp_dot[2][1]]=1
  #temp_pred[temp_dot[3][1]]=1
  #print(df.iloc[row_num].values[2:13])
  #break
  #print(np.dot(np.asarray(df.iloc[row_num].values[2:13]), np.asarray(temp_pred)))

  # Number of clusters
  kmeans = KMeans(n_clusters=2)
  # Fitting the input data
  kmeans = kmeans.fit(temp_dot.reshape(-1, 1))
  # Getting the cluster labels
  labels = kmeans.predict(temp_dot.reshape(-1, 1))
  #print(temp_dot)
  #print(labels)
  # Centroid values
  centroids = kmeans.cluster_centers_
  #print(centroids)
  break_count += 1
  #if(break_count ==5):
  #  break
  
  count = 0
  if(centroids[0] < centroids[1]):
    for idxx, ele in enumerate(labels):
      if(ele == df.iloc[row_num].values[2+idxx]):
        count += 1
  else:
    for idxx, ele in enumerate(labels):
      if(ele != df.iloc[row_num].values[2+idxx]):
        count += 1
  
  total_count += count
  
print(total_count/(886*11))

'''
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
'''
