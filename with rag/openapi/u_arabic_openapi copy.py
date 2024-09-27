import streamlit as st
import openai
import fitz  # PyMuPDF for PDF reading
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from arabic_support import support_arabic_text

support_arabic_text(all=True)

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

# Custom CSS for enhanced Arabian-inspired chatbot style with proper RTL support
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri&family=Cairo:wght@600&display=swap');

    body {
        background-image: url('https://www.publicdomainpictures.net/pictures/320000/velka/plain-golden-pattern-background.jpg');
        background-size: cover;
        font-family: 'Amiri', serif;
    }

    .css-18e3th9 {
        font-family: 'Cairo', sans-serif;
    }

    .stButton button {
        background-color: #cc9966;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 18px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #b38b5b;
    }

    .stTextInput input {
        background-color: #f9f5ec;
        border: 2px solid #cc9966;
        border-radius: 10px;
        padding: 12px;
        color: #333333;
        font-size: 18px;
        text-align: right;  /* Align text to the right for Arabic */
    }
    .stTextInput label {
        color: #333333;
        font-size: 18px;
        text-align: right;  /* Align label to the right for Arabic */
    }

    .chat-bubble {
        background-color: #f4e3d7;
        border-radius: 20px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .user-chat {
        background-color: #cc9966;
        color: white;
        text-align: right;
        margin-left: auto;
    }
    .assistant-chat {
        background-color: #f4e3d7;
        color: #333333;
        text-align: right;
        margin-right: auto;
    }

    h1 {
        color: #cc9966;
        font-family: 'Cairo', sans-serif;
        text-align: center;
    }
    footer {
        font-size: 14px;
        text-align: center;
        color: #777;
        padding: 10px 0;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app setup
st.title("PDF مساعد ")
st.write("قم برفع ملفات PDF اسأل أسئلة بناءً على محتواها")

# File uploader for PDFs (multiple file support)
uploaded_pdfs = st.file_uploader("قم برفع ملفات", type="pdf", accept_multiple_files=True)

if uploaded_pdfs:
    with st.spinner("...جاري استخراج النص من ملفات "):
        pdf_text = extract_text_from_pdfs(uploaded_pdfs)
    
    st.success("! تم استخراج النصوص بنجاح")

    # Split the text into manageable chunks
    chunks = split_text_into_chunks(pdf_text)

    # Display a text input for asking questions
    question = st.text_input("اكتب سؤالك هنا")

    if st.button("إرسال"):
        if question:
            with st.spinner("...جاري البحث عن المحتوى المناسب"):
                relevant_chunks = find_relevant_chunks(question, chunks)
                answer = ask_openai_question(question, relevant_chunks)

            # Display the question and answer as chat bubbles
            st.markdown(f"""
            <div class="chat-bubble user-chat">**سؤالك:** {question}</div>
            <div class="chat-bubble assistant-chat">**الإجابة:** {answer}</div>
            """, unsafe_allow_html=True)
        else:
            st.error("الرجاء إدخال سؤال")

# Footer
st.markdown("""
    <div style='text-align: center; padding: 10px; font-size: 14px; color: #777; margin-top: 50px;'>
        © 2024 مساعد الـ PDF - جميع الحقوق محفوظة
    </div>
""", unsafe_allow_html=True)