import openai

# Set your OpenAI API key
openai.api_key = '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'
def ask_openai_question(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can use 'gpt-4' if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            max_tokens=100,  # Limit the response length
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except openai.error.OpenAIError as e:
        return f"Error: {e}"

# Example of a simple question
question = "What is the capital of France?"

# Call the function and print the response
answer = ask_openai_question(question)
print(f"Question: {question}")
print(f"Answer: {answer}")