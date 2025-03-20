import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
#sample data
data = {
    "plot": [
        "A spaceship crew explores an uncharted planet and encounters aliens.",
        "A detective investigates a murder in a small town.",
        "A young wizard learns magic at a school of witchcraft.",
        "A team of superheroes save the world from an evil villain.",
        "A man and woman fall in love despite their families' objections."
    ],
    "genre": ["Sci-Fi", "Mystery", "Fantasy", "Action", "Romance"]
}

df = pd.DataFrame(data)

# Text preprocessing and feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['plot'])
y = df['genre']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Interactive loop for real-time predictions
while True:
    new_plot = input("Enter a movie plot (or type 'exit' to quit): ")
    if new_plot.lower() == 'exit':
        break
    new_X = vectorizer.transform([new_plot])
    predicted_genre = classifier.predict(new_X)
    print(f"Predicted Genre: {predicted_genre[0]}")
