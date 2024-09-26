import streamlit as st
import openai
import fitz  # PyMuPDF for PDF reading
import io

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

# Function to ask OpenAI a question based on the PDF content
def ask_openai_question(question, context):
    prompt = f"Here is the content of a document:\n{context}\n\nBased on this, answer the following question:\n{question}"
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

    # Display a text input for asking questions
    question = st.text_input("Ask a question based on the PDF:")
    
    if st.button("Ask"):
        if question:
            with st.spinner("Asking OpenAI..."):
                answer = ask_openai_question(question, pdf_text)
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Answer:** {answer}")
        else:
            st.error("Please enter a question.")

