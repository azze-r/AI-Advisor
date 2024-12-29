import streamlit as st
import openai
import fitz  # PyMuPDF for PDF reading
import io
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from arabic_support import support_arabic_text
from tiktoken import encoding_for_model  # For estimating token count
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Enable Arabic text support
support_arabic_text(all=True)

# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_stream = io.BytesIO(pdf_file.read())
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    for page in doc:
        page_text = page.get_text()
        # Sanitize the text by replacing all 10-digit numbers with "xxxxxxxxxx"
        sanitized_text = re.sub(r'\b\d{8,14}\b', 'xxxxxxxxxx', page_text)
        text += sanitized_text
    return text

# Function to extract text from multiple PDFs
def extract_text_from_pdfs(pdf_files):
    all_text = ""
    for pdf_file in pdf_files:
        all_text += extract_text_from_pdf(pdf_file) + "\n\n"
    
    return all_text

# Function to anonymize legal text
def anonymize_legal_text(text):
    # Load the NER model and tokenizer
    model_name = "marefa-nlp/marefa-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    # Create a pipeline for NER
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Use the NER pipeline to extract named entities
    ner_results = ner_pipeline(text)

    # Initialize positions and anonymized version of text
    anonymized_text = text
    offset = 0

    for entity in ner_results:
        if "person" in entity['entity_group']:  # Focus specifically on person names
            start = entity['start']
            end = entity['end']
            # Adjust start and end for cumulative changes made above
            adjusted_start = start + offset
            adjusted_end = end + offset
            anonymized_text = (anonymized_text[:adjusted_start] + 
                               '******' + 
                               anonymized_text[adjusted_end:])
            # Update offset due to increase in string length difference
            offset += len('******') - (end - start)

    return anonymized_text

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
    prompt = f"Here is the content of a document:\n{context}\n\nBased on this, answer the following question in Arabic but never tell any sensitive information (like name, phone number, number card ...) :\n{question}"

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except openai.error.AuthenticationError:
        return "مفتاح OpenAI API غير صحيح. يرجى التأكد من صحته."
    except openai.error.InvalidRequestError as e:
        return f"Error: {e}"

# Function to process the user's question and retrieve relevant context
def process_question(question):
    # Retrieve relevant context from conversation history
    context = st.session_state.context.get(question, "")
    return context


# Initialize session state for OpenAI API key, conversation history, context, and question input
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if 'conversation' not in st.session_state:
    st.session_state.conversation = []  # Initialize conversation history

if 'context' not in st.session_state:
    st.session_state.context = {}  # Initialize context storage

if 'question_input' not in st.session_state:
    st.session_state.question_input = ""  # Initialize question input

# Input field for OpenAI API key
st.markdown("<h1 style='text-align: center; color: #006400;'>مساعد PDF الذكي</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>يرجى إدخال مفتاح API الخاص بك لـ OpenAI للمتابعة.</p>", unsafe_allow_html=True)
api_key_input = st.text_input("🔑 أدخل مفتاح OpenAI API الخاص بك هنا:", type="password")


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

        # Anonymize the text (run only once when PDFs are uploaded)
        if 'anonymized_text' not in st.session_state:
            st.session_state.anonymized_text = anonymize_legal_text(pdf_text)
            st.success("✅ تم تعديل النصوص بنجاح!")

        # Split the text into manageable chunks
        chunks = split_text_into_chunks(st.session_state.anonymized_text)

        # Initialize a container to dynamically add question-answer pairs
        conversation_container = st.container()

        # Move the question input box to the sidebar
        st.sidebar.markdown("<h3>💬 اكتب سؤالك هنا:</h3>", unsafe_allow_html=True)
        question_input = st.sidebar.text_input("اكتب سؤالك بالعربية", key="text_input", value=st.session_state.question_input)

        if st.sidebar.button("إرسال"):
            if question_input:
                # Add user's question to conversation history
                st.session_state.conversation.append({"role": "user", "content": question_input})

                # Retrieve context from conversation history
                context = process_question(question_input)

                # Limit the number of chunks to ensure we don't exceed token limit
                relevant_chunks = limit_chunks_by_tokens(chunks, max_tokens=16000)

                # Ask OpenAI for an answer
                with st.spinner("🔍 جاري البحث عن المحتوى المناسب..."):
                    answer = ask_openai_question(question_input, relevant_chunks, st.session_state.api_key)

                # Add the assistant's response to the conversation history and context
                st.session_state.conversation.append({"role": "assistant", "content": answer})
                st.session_state.context[question_input] = context

                # Display the conversation dynamically
                with conversation_container:
                    st.markdown("<h3>📝 المحادثة</h3>", unsafe_allow_html=True)
                    for message in st.session_state.conversation:
                        if message["role"] == "user":
                            st.markdown(f"""
                            <div class='chat-bubble user'>
                                <b>🧑‍💼 أنت:</b> {message['content']}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='chat-bubble assistant'>
                                <b>🤖 المساعد:</b> {message['content']}
                            </div>
                            """, unsafe_allow_html=True)

# Custom CSS for enhanced Arabian-inspired chatbot style with proper RTL support
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Almarai&family=Noto+Sans+Arabic&display=swap');
    
    body {
        background: linear-gradient(45deg, #006400, #8B4513);  /* Green and brown */
        font-family: 'Almarai', sans-serif;
        color: white;
    }
    
    .chat-bubble {
        background-color: #f4e3d7;
        border-radius: 20px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .chat-bubble.user {
        background-color: #e8f5e9;
        text-align: right;
    }
    
    .chat-bubble.assistant {
        background-color: #c8e6c9;
        text-align: right;
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
        text-align: right;
    }
    
    .stTextInput label {
        color: #333333;
        font-size: 18px;
        text-align: right;
    }
    
    </style>
""", unsafe_allow_html=True)