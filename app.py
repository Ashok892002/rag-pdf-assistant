import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

# API Key
os.environ["GROQ_API_KEY"] = "paste_your_groq_key_here"

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def get_answer(vector_store, question):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context only. If answer is not in context, say 'I could not find this in the document.'"},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    response = llm.invoke(messages)
    return response.content

def main():
    # Page config
    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stApp {max-width: 900px; margin: auto;}
    .title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5em;
        font-weight: bold;
        padding: 20px;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .success-box {
        background-color: #d5f5e3;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="title">🤖 RAG-Powered Personal Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload any PDF and chat with it using AI!</div>', unsafe_allow_html=True)

    # Divider
    st.divider()

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png")
        st.title("📁 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        st.divider()
        st.markdown("### ℹ️ How to use:")
        st.markdown("1. Upload a PDF file")
        st.markdown("2. Wait for processing")
        st.markdown("3. Ask any question!")
        st.divider()
        st.markdown("### 🛠️ Built With:")
        st.markdown("- 🦜 LangChain")
        st.markdown("- 🤗 HuggingFace")
        st.markdown("- ⚡ Groq LLaMA3")
        st.markdown("- 📊 FAISS")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Process PDF
    if uploaded_file is not None:
        with st.spinner("📖 Reading and processing your PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            st.session_state.vector_store = create_vector_store(text)

        st.success(f"✅ **{uploaded_file.name}** processed successfully!")
        st.info("💬 Ask me anything about your document below!")

        st.divider()

        # Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Question Input
        question = st.chat_input("💬 Type your question here...")

        if question:
            with st.chat_message("user"):
                st.write(question)
            st.session_state.messages.append({"role": "user", "content": question})

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    answer = get_answer(st.session_state.vector_store, question)
                st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        # Welcome screen
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📄 **Step 1**\nUpload your PDF from the sidebar")
        with col2:
            st.info("🧠 **Step 2**\nAI processes your document")
        with col3:
            st.info("💬 **Step 3**\nAsk anything about it!")

if __name__ == "__main__":
    main()