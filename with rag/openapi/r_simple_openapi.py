import streamlit as st
import openai
import fitz  # PyMuPDF for PDF reading
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set your OpenAI API key
openai.api_key = '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'

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_stream = io.BytesIO(pdf_file.read())
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# Function to split text into smaller chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Function to find the most relevant chunks
def find_relevant_chunks(question, chunks):
    # Simple method: find chunks that contain key terms from the question
    relevant_chunks = [chunk for chunk in chunks if any(term in chunk.lower() for term in question.lower().split())]
    return relevant_chunks if relevant_chunks else chunks[:3]  # Fallback: return first 3 chunks if no relevant chunks found

# Function to ask OpenAI a question based on the most relevant chunks of the PDF
def ask_openai_question(question, context_chunks):
    # Join the chunks into context
    context = "\n\n".join(context_chunks)
    prompt = f"Here is the content of a document:\n{context}\n\nBased on this, answer the following question:\n{question}"
    
    # Ensure the prompt stays within token limits (adjusting max_tokens based on the model limit)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can use 'gpt-4' if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()

# Streamlit app setup
st.title("PDF Q&A with OpenAI")
st.write("Upload a PDF file and ask questions based on its content.")

# File uploader for PDFs
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_pdf)
    
    st.success("Text extracted from the PDF successfully!")

    # Split the text into manageable chunks
    chunks = split_text_into_chunks(pdf_text)

    # Display a text input for asking questions
    question = st.text_input("Ask a question based on the PDF:")
    
    if st.button("Ask"):
        if question:
            with st.spinner("Finding relevant content and asking OpenAI..."):
                # Find the most relevant chunks for the question
                relevant_chunks = find_relevant_chunks(question, chunks)
                answer = ask_openai_question(question, relevant_chunks)
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Answer:** {answer}")
        else:
            st.error("Please enter a question.")
