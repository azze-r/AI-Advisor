from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import fitz  # PyMuPDF
from langchain.docstore.document import Document

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

# Specify the path to your PDF file
pdf_path = "C:\\Users\\a813538\\Downloads\\RAGGemmaModel-main 1\\RAGGemmaModel-main\\mental_health_Document.pdf"

print("Extracting text from PDF and splitting it")
pdf_text = extract_text_from_pdf(pdf_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_texts = text_splitter.split_text(pdf_text)

print("Embedding vector model")
oembed = OllamaEmbeddings(model="nomic-embed-text")

documents = [Document(page_content=split) for split in split_texts]

print("Storing it to Chroma")
vectorstore = Chroma.from_documents(documents=documents, embedding=oembed)

print("Asking questions on this vectorized model")
question = "Who is Neleus and who is in Neleus' family?"
docs = vectorstore.similarity_search(question)

print("Displaying number of occurrences")
print(len(docs))