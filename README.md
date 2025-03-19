This repository contains three different machine learning models:

Movie Genre Classification - Predicts the genre of a movie based on its plot summary.

Customer Churn Prediction - Predicts whether a customer will churn based on their usage and demographic data.

Handwritten-Like Text Generation - Generates text based on a given seed using a Markov-based approach.

1. Movie Genre Classification

Description

This model predicts the genre of a movie based on a given plot summary. It uses the TF-IDF vectorization technique and a Logistic Regression classifier.

Requirements

Python 3.x

numpy

pandas

sklearn

Usage

Run the script and enter a movie plot when prompted:

python movie_genre_classifier.py

Enter a plot description, and the model will output the predicted genre.

2. Customer Churn Prediction

Description

This model predicts whether a customer will churn based on features like monthly usage, customer age, and subscription length. It uses a Random Forest Classifier.

Requirements

Python 3.x

numpy

pandas

sklearn

Usage

Run the script and input customer details when prompted:

python customer_churn_prediction.py

Enter the required customer details, and the model will predict if the customer is likely to churn.

3. Handwritten-Like Text Generation

Description

This model generates text using a Markov-based approach that selects characters randomly based on prior sequences.

Requirements

Python 3.x

numpy

Usage

Run the script and enter a seed text when prompted:

python text_generation.py

The model will generate new text based on the provided input.

Notes

Each script runs interactively in the terminal.

Datasets should be replaced with real-world data for better accuracy.

The text generation model can be improved by incorporating deep learning techniques like LSTMs.
