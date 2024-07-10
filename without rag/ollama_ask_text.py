import ollama
response = ollama.chat(model='gemma', messages=[
  {
    'role': 'user',
    'content': """
     hello
""",
  },
])
print(response['message']['content'])