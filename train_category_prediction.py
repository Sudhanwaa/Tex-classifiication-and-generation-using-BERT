import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.utils import to_categorical
import pickle

# Loading the dataset
df1 =pd.read_csv("train.csv")
df2 =pd.read_csv("test.csv")
df1 = df1.dropna(subset=['Summary'])
print(df1["Summary"].isnull().sum())
df1.shape
training_df=df1
testing_df=df2
training_df= training_df.dropna(subset=['Summary'])
training_df['Summary'].isnull().sum()

# Loading the encoder and preprocessing models for BERT
encoder_url="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
preprocess_url="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

bert_encoder=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
bert_preprocess=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

# Categorically encoding the classes(labels)
encoder = LabelEncoder()
training_df['class'] = encoder.fit_transform(training_df['Category'])
mapping = dict(zip(training_df['Category'], training_df['class']))
testing_df['class'] = testing_df['Category'].map(mapping)


training_df['Summary']  =training_df['Summary'] .str.replace('[^a-zA-Z]', ' ')

# Dividing into training dataset and labels
X=training_df['Summary']
Y=training_df['class']

# Assuming your original labels are integers ranging from 0 to 41
one_hot_labels = to_categorical(Y, num_classes=42)
null_count = Y.isnull().sum()
null_count
one_hot_labels.shape

# Define the input layer
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text) 


x = tf.keras.layers.Dropout(0.3, name="dropout")(outputs['pooled_output'])
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(128, activation='sigmoid')(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Output layer
# 42 classes for 42 labels
output = tf.keras.layers.Dense(42, activation='softmax', name='output')(x)
model = tf.keras.Model(inputs=[text_input], outputs=[output])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compiling and training the model 
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'] )
model.fit(X,one_hot_labels,epochs=5)

