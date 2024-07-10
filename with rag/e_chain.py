from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    # Ouvrir le document PDF
    doc = fitz.open(pdf_path)
    text = ""
    # Parcourir chaque page du document PDF
    for page in doc:
        text += page.get_text()
    return text

# Spécifiez le chemin vers votre fichier PDF - B
pdf_pathA = "C:\\Users\\a813538\\Downloads\\RAGGemmaModel-main 1\\RAGGemmaModel-main\\cv.pdf"

print("Extraire le texte du fichier PDF et splitter")
pdf_textA = extract_text_from_pdf(pdf_pathA)

# Chunk big pdf text into small parts of text
print("chunk")
text_splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=200)
all_splits = text_splitter.split_text(pdf_textA)
# Convert splits to Document objects
documents = [Document(page_content=split) for split in all_splits]

print("embedd vector model")
# Nous allons convertir le text splité en format vectoriel en utilisant
# le modele ollama nomé "nomic-embed-text" qui tourne lui aussi en local
#embedding
oembed = OllamaEmbeddings(model="nomic-embed-text")

print("store it to chroma")
# Une fois le nouveau format généré nommé embeddings, nous allons le stocker
# dans une base de connaissance pour l'exemple Chroma
# vector database + relevant data 
vectorstore = Chroma.from_documents(documents=documents, embedding=oembed)

print("load our model")
modelChoiced = "gemma2"
print("Appel au Modèle")
ollama = Ollama(
    model=modelChoiced
)

print("Chain between vectorstore and model")
# Appel à l'LLM (Retrieval Augmented Generation)
qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
question="Does this profile match for android job"
print(qachain.invoke({"query": question}))