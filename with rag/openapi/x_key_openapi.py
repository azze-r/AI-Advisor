import streamlit as st
import openai
import fitz  # PyMuPDF for PDF reading
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from arabic_support import support_arabic_text
from tiktoken import encoding_for_model  # For estimating token count

# Enable Arabic text support
support_arabic_text(all=True)

# Initialize session state for OpenAI API key and conversation history
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if 'conversation' not in st.session_state:
    st.session_state.conversation = []  # Initialize conversation history

# Input field for OpenAI API key
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>مساعد PDF الذكي</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>يرجى إدخال مفتاح API الخاص بك لـ OpenAI للمتابعة.</p>", unsafe_allow_html=True)
api_key_input = st.text_input("🔑 أدخل مفتاح OpenAI API الخاص بك هنا:", type="password")

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

# Function to estimate token count for a given text
def estimate_token_count(text, model="gpt-3.5-turbo"):
    encoding = encoding_for_model(model)
    return len(encoding.encode(text))

# Function to limit the number of chunks based on token limits
def limit_chunks_by_tokens(chunks, max_tokens=4000, model="gpt-3.5-turbo"):
    total_tokens = 0
    limited_chunks = []
    
    for chunk in chunks:
        chunk_tokens = estimate_token_count(chunk, model=model)
        if total_tokens + chunk_tokens <= max_tokens:
            limited_chunks.append(chunk)
            total_tokens += chunk_tokens
        else:
            break
    
    return limited_chunks

# Function to ask OpenAI a question based on the most relevant chunks of the PDFs
def ask_openai_question(question, context_chunks, api_key, model="gpt-3.5-turbo", max_tokens=300):
    openai.api_key = api_key
    context = "\n\n".join(context_chunks)
    prompt = f"Here is the content of a document:\n{context}\n\nBased on this, answer the following question:\n{question}"
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except openai.error.AuthenticationError:
        return "مفتاح OpenAI API غير صحيح. يرجى التأكد من صحته."
    except openai.error.InvalidRequestError as e:
        return f"Error: {e}"

# Store the API key once the user submits it
if api_key_input:
    st.session_state.api_key = api_key_input
    st.success("تم إدخال مفتاح API بنجاح! ✅")

# Check if an API key is provided before proceeding
if st.session_state.api_key:
    # File uploader for PDFs (multiple file support)
    uploaded_pdfs = st.file_uploader("📄 قم برفع ملفات PDF", type="pdf", accept_multiple_files=True)

    if uploaded_pdfs:
        with st.spinner("⏳ جاري استخراج النص من ملفات PDF..."):
            pdf_text = extract_text_from_pdfs(uploaded_pdfs)

        st.success("✅ تم استخراج النصوص بنجاح!")

        # Split the text into manageable chunks
        chunks = split_text_into_chunks(pdf_text)

        # Display a text input for asking questions
        question = st.text_input("💬 اكتب سؤالك هنا:")

        if st.button("إرسال"):
            if question:
                # Add user's question to the conversation history
                st.session_state.conversation.append({"role": "user", "content": question})

                # Limit the number of chunks to ensure we don't exceed token limit
                relevant_chunks = limit_chunks_by_tokens(chunks, max_tokens=16000)

                # Ask OpenAI for an answer
                with st.spinner("🔍 جاري البحث عن المحتوى المناسب..."):
                    answer = ask_openai_question(question, relevant_chunks, st.session_state.api_key)

                # Add the assistant's response to the conversation history
                st.session_state.conversation.append({"role": "assistant", "content": answer})

# Display the conversation history in a chatbot-like format
st.markdown("<h3>📝 المحادثة</h3>", unsafe_allow_html=True)
for message in st.session_state.get('conversation', []):
    if message["role"] == "user":
        st.markdown(f"""
            <div style='background-color: #e1f3fb; border-radius: 10px; padding: 10px; margin: 10px 0; text-align: left;'>
                <b>🧑‍💼 أنت:</b> {message['content']}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='background-color: #d4f8e8; border-radius: 10px; padding: 10px; margin: 10px 0; text-align: left;'>
                <b>🤖 المساعد:</b> {message['content']}
            </div>
        """, unsafe_allow_html=True)

# Custom CSS for chatbot bubbles
st.markdown("""
    <style>
        .stTextInput, .stButton {
            margin-top: 20px;
        }
        div[data-testid="stTextInput"] {
            width: 100%;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; padding: 10px; font-size: 14px; color: #777; margin-top: 50px;'>
        © 2024 مساعد الـ PDF - جميع الحقوق محفوظة
    </div>
""", unsafe_allow_html=True)
