from langchain_community.llms import Ollama

# Initialize the Ollama client
ollama = Ollama(
    model="gemma"  # Specify the correct model name
)

# Example usage
answer = ollama.invoke("hello")
print(answer)