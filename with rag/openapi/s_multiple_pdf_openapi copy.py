import streamlit as st
import openai
import fitz  # PyMuPDF for PDF reading
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set your OpenAI API key
openai.api_key = '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'

# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_stream = io.BytesIO(pdf_file.read())
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# Function to extract text from multiple PDFs
def extract_text_from_pdfs(pdf_files):
    all_text = ""
    for pdf_file in pdf_files:
        all_text += extract_text_from_pdf(pdf_file) + "\n\n"
    return all_text

# Function to split text into smaller chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Function to find the most relevant chunks
def find_relevant_chunks(question, chunks):
    relevant_chunks = [chunk for chunk in chunks if any(term in chunk.lower() for term in question.lower().split())]
    return relevant_chunks if relevant_chunks else chunks[:3]  # Fallback: return first 3 chunks if no relevant chunks found

# Function to ask OpenAI a question based on the most relevant chunks of the PDFs
def ask_openai_question(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"Here is the content of a document:\n{context}\n\nBased on this, answer the following question:\n{question}"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()

# Custom CSS for Arabian-inspired chatbot style
st.markdown("""
    <style>
    body {
        background-color: #f4f0e6;
        font-family: 'Amiri', serif;
    }
    .css-18e3th9 {
        font-family: 'Amiri', serif;
    }
    .stButton button {
        background-color: #cc9966;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
    }
    .stTextInput input {
        background-color: #f4f0e6;
        border: 2px solid #cc9966;
        border-radius: 8px;
        padding: 10px;
        color: #333333;
    }
    .stTextInput label {
        color: #333333;
    }
    .stMarkdown h1 {
        color: #cc9966;
        text-align: center;
    }
    .stMarkdown h2 {
        color: #cc9966;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app setup
st.title("مساعد الـ PDF")
st.write("قم بتحميل ملفات PDF واسأل عن محتواها.")

# File uploader for PDFs (multiple file support)
uploaded_pdfs = st.file_uploader("قم بتحميل ملفات PDF", type="pdf", accept_multiple_files=True)

if uploaded_pdfs:
    with st.spinner("استخراج النصوص من ملفات PDF..."):
        pdf_text = extract_text_from_pdfs(uploaded_pdfs)
    
    st.success("تم استخراج النصوص بنجاح!")

    # Split the text into manageable chunks
    chunks = split_text_into_chunks(pdf_text)

    # Display a text input for asking questions
    question = st.text_input("اسأل سؤالك هنا:")

    if st.button("إرسال"):
        if question:
            with st.spinner("جاري البحث عن المحتوى المناسب..."):
                relevant_chunks = find_relevant_chunks(question, chunks)
                answer = ask_openai_question(question, relevant_chunks)
            st.markdown(f"**السؤال:** {question}")
            st.markdown(f"**الإجابة:** {answer}")
        else:
            st.error("الرجاء إدخال سؤال.")
