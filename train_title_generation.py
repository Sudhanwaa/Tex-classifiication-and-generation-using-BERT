import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
import pickle 
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

df1 =pd.read_csv("train.csv")
df2 =pd.read_csv("test.csv")
df1 = df1.dropna(subset=['Summary'])
print(df1["Summary"].isnull().sum())
df1.shape
################### MODEL 1 ###########################################

encoder_url="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
preprocess_url="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

bert_encoder=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
bert_preprocess=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")



training_df=df1
testing_df=df2

training_df= training_df.dropna(subset=['Summary'])
training_df = training_df.dropna(subset=['Headline'])
training_df['Headline'].isnull().sum()
# training_df['Summary'].isnull().sum()

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation and lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Apply preprocessing to the "headline" column
training_df['headline_tokens'] = training_df['Headline'].apply(preprocess_text)

# Train Word2Vec model
model = Word2Vec(sentences=training_df['headline_tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Embed the "headline" column
def embed_text(text):
    # Sum the vectors of individual words
    embeddings = [model.wv[word] for word in text if word in model.wv]
    if embeddings:
        return sum(embeddings)
    else:
        return None

# Apply embedding to the "headline" column
training_df['headline_embedding'] = training_df['headline_tokens'].apply(embed_text)

training_df = training_df.dropna(subset=['headline_embedding'])

embedding_series = training_df["headline_embedding"]

# Convert Pandas Series to a list of NumPy arrays
embedding_list = []

for embedding in embedding_series:
    if isinstance(embedding, (list, np.ndarray)):
        # Ensure that all elements are 1-dimensional arrays or lists
        if np.array(embedding).ndim == 1:
            embedding_list.append(embedding)
        else:
            # If it's a multi-dimensional array, you may want to flatten it or handle it accordingly
            # Here, I'm assuming you want to flatten it
            embedding_list.extend(embedding)
    elif isinstance(embedding, (int, float)):
        # Handle single numeric values if needed
        embedding_list.append([embedding])
    else:
      
        pass

# Convert the list of NumPy arrays to a TensorFlow tensor
embedding_tensor = tf.convert_to_tensor(embedding_list, dtype=tf.float32)

# Checking the Dimensions of training and testing dataset
print(training_df["headline_embedding"].shape)
print("Tensor->",embedding_tensor.shape)
training_df["Summary"].shape

# Define the input layer
num_unique_headlines = len(set(training_df['Headline']))
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Add additional layers
x = tf.keras.layers.Dropout(0.3, name="dropout")(outputs['pooled_output'])
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Output layer
output = tf.keras.layers.Dense(units =100, activation='linear', name='output')(x)

model = tf.keras.Model(inputs=[text_input], outputs=[output])

#Training and compilation 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'] )
model.fit(training_df["Summary"],embedding_tensor, epochs=3)

# Saving the model
with open('headline_prediction.pkl', 'wb') as file:
    pickle.dump(model, file)

################################################################################################################

# ############################################# MODEL 2 ########################################################



# Load GPT2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Input a test Summary
summary = "Accidentally put grown-up toothpaste on my toddler’s toothbrush and he screamed like I was cleaning his teeth with a Carolina Reaper dipped in Tabasco sauce."

# Tokenize inputted summary
input_ids = tokenizer.encode(summary, return_tensors="pt")

# Generate headline using the GPT-2 model
output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

# Decoding headline
generated_headline = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated headline
print("Generated Headline:", generated_headline)

#####################################################################################################3########3#
