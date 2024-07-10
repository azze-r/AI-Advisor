from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
data = loader.load()
print(data)