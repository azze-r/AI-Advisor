from transformers import pipeline

# Load the pre-trained sentiment analysis model
sentiment_analysis = pipeline(
"sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

input_text = [
"It’s a great app, my biggest problem is the card readers regularly do not connect. Which is very poor customer service for us because we have to manually enter our customers debit cards, which takes time. This slows down our efficiency."
]

# Perform sentiment analysis on the input text
result = sentiment_analysis(input_text)

# Print the result
print(result)