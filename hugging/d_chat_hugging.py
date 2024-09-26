# Load libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Instantiate tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Define input text and apply tokenizer
input_text = "The square root of x is the cube root of y. What is y to the power of 2, if x = 4?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Run the model
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))