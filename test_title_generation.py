import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df1 =pd.read_csv("train.csv")
df2 =pd.read_csv("test.csv")
df1 = df1.dropna(subset=['Summary'])


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
        # Handle other cases accordingly
        pass

# Convert the list of NumPy arrays to a TensorFlow tensor
embedding_tensor = tf.convert_to_tensor(embedding_list, dtype=tf.float32)

path="headline_generation.pkl"
with open(path, 'rb') as file:
    loaded_model = pickle.load(file)

loaded_model.evaluate(testing_df["Summary"],embedding_list)