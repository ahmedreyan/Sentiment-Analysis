import re
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the dataset
data = pd.read_csv("booking_reviews copy.csv")

# Function to preprocess the text
def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    return text

# Preprocess the text data
data['processed_review'] = data['review_text'].apply(preprocess)

# Command-line interface for continuous user input
while True:
    user_input = input("Enter a review (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break

    # Preprocess the user input
    processed_input = preprocess(user_input)

    # Tokenize the input
    encoded_input = tokenizer.encode_plus(processed_input, return_tensors='pt')

    # Make a prediction
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Get the sentiment with the highest score
    sentiment_labels = ['negative', 'neutral', 'positive']
    predicted_sentiment = sentiment_labels[scores.argmax()]

    # Print the predicted sentiment
    print(f"Predicted sentiment: {predicted_sentiment}")