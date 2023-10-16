import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
training_df['class'] = encoder.fit_transform(training_df['Category'])
mapping = dict(zip(training_df['Category'], training_df['class']))
#   Did this so that the classes have the same encoded labels in the training as well as testing set
testing_df['class'] = testing_df['Category'].map(mapping)

testing_df= testing_df.dropna(subset=['Summary'])
testing_df['Summary'].isnull().sum()
one_hot_labels = to_categorical(testing_df["class"], num_classes=42)


path="category_prediction.pkl"
with open(path, 'rb') as file:
    loaded_model = pickle.load(file)

loaded_model.evaluate(testing_df["Summary"],one_hot_labels)