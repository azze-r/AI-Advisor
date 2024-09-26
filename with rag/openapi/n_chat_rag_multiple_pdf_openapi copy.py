import streamlit as st
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import fitz  # PyMuPDF
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API key setup (use your own API key or set it as an environment variable)
openai.api_key = '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'
# Function to extract text from multiple PDFs
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        logger.info(f"Extracting text from PDF file: {pdf_file.name}")
        # Convert the Streamlit UploadedFile to BytesIO
        pdf_stream = io.BytesIO(pdf_file.read())
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        for page in doc:
            text += page.get_text()
    logger.info("Extracted text from all PDF files")
    return text

# Function to generate embeddings using OpenAI's text-embedding-ada-002
def generate_openai_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
        embedding = response['data'][0]['embedding']
        embeddings.append(embedding)
    return embeddings

# Function to ask OpenAI a question based on retrieved document chunks
def ask_openai_question(question, context):
    prompt = f"Here is some context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",  # or 'gpt-4' if you have access
        prompt=prompt,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].text.strip()

@st.cache_resource
def load_model_and_prepare_qa(pdf_files):
    # Extract text from the uploaded PDF files
    pdf_text = extract_text_from_pdfs(pdf_files)

    # Split the text into chunks
    logger.info("Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=200)
    all_splits = text_splitter.split_text(pdf_text)

    # Generate embeddings using OpenAI
    logger.info("Generating embeddings with OpenAI")
    embeddings = generate_openai_embeddings(all_splits)

    # Convert splits to Document objects
    logger.info("Converting text splits to Document objects")
    documents = [Document(page_content=split) for split in all_splits]

    # Store embeddings in Chroma vectorstore
    logger.info("Storing embeddings in Chroma vectorstore")
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)

    # Define a QA system where you will query OpenAI using the retrieved context
    def qa_system(question):
        # Retrieve relevant documents from vectorstore
        logger.info(f"Retrieving relevant documents for question: {question}")
        relevant_documents = vectorstore.similarity_search(question, k=3)  # Retrieve top 3 chunks

        # Prepare the context for OpenAI
        context = "\n\n".join([doc.page_content for doc in relevant_documents])

        # Ask the question to OpenAI
        logger.info("Asking OpenAI the question")
        answer = ask_openai_question(question, context)
        return answer

    logger.info("QA system ready")
    return qa_system

# Streamlit UI components
st.set_page_config(page_title="Document-based Chatbot with RAG", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #2e3b4e;
        color: white;
    }
    .st-bt {
        background-color: #007BFF;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 Document-based Chatbot with RAG")
st.markdown("### Upload your PDFs and ask questions about their content")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# File uploader for the PDFs
uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Load the model and prepare the QA system
    with st.spinner("Loading model and preparing QA system..."):
        logger.info("Loading model and preparing QA system")
        qa_system = load_model_and_prepare_qa(uploaded_files)

    st.success("Model loaded and QA system ready!")
    logger.info("Model loaded and QA system ready")

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Bot:** {chat['answer']}")

    # Input box for user questions
    st.markdown("### Ask a question about the documents:")
    question = st.text_input("Your question:", key="question_input")

    if st.button("Ask"):
        with st.spinner("Processing your question..."):
            logger.info(f"Processing question: {question}")
            answer = qa_system(question)
        st.markdown(f"**You:** {question}")
        st.markdown(f"**Bot:** {answer}")
        logger.info(f"Answer: {answer}")

        # Update chat history in session state
        st.session_state.chat_history.append({"question": question, "answer": answer})

# Footer
st.markdown(
    """
    <div style="text-align: center; padding: 20px;">
        Made with ❤️ by [Your Name]
    </div>
    """,
    unsafe_allow_html=True
)
