import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="Swiggy AI Assistant", layout="wide")
st.title("🤖 Swiggy AI Assistant")

# --- Sidebar ---
with st.sidebar:
    st.header("Setup")
    groq_key = st.secrets.get("GROQ_API_KEY")
    uploaded_file = st.file_uploader("Upload Swiggy Report PDF", type="pdf")
    process_btn = st.button("Build System")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Core RAG Logic ---
def setup_rag(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()


    
    # Improved Chunking for Financial Data
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "] # Priority for splitting
    )
    chunks = splitter.split_documents(data)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Increase k to 5 for better information density
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Get key from secrets automatically
# Internal secret fetch
    llm = ChatGroq(
        temperature=0.1, 
        groq_api_key=st.secrets["GROQ_API_KEY"], 
        model_name="llama-3.3-70b-versatile"
    )
    # ... rest of the logic ...
    return rag_chain, retriever
    # Professional Prompt Template
    template = """You are a Financial Analyst Assistant for Swiggy. 
    Use the retrieved context to provide a precise, data-driven answer.
    If the context contains tables, interpret them carefully.
    If you cannot find the specific data point, state: "The report does not provide this specific information."
    
    Context: {context}
    Question: {question}
    Answer:"""


# --- Execution ---
if process_btn and uploaded_file and groq_key:
    with st.spinner("Processing PDF..."):
        with open("temp_report.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        
        chain, retriever = setup_rag("temp_report.pdf")
        st.session_state.rag_chain = chain
        st.session_state.retriever = retriever
        st.success("Ready!")

query = st.text_input("Ask about Swiggy's FY 23-24 performance:")

if query and "rag_chain" in st.session_state:
    try:
        result = st.session_state.rag_chain.invoke(query)
        st.subheader("Answer:")
        st.write(result)
        
        docs = st.session_state.retriever.invoke(query)
        with st.expander("Show Evidence from Report"):
            for doc in docs:
                st.markdown(f"**Page {doc.metadata.get('page', '?')}:**")
                st.info(doc.page_content)
    except Exception as e:
        st.error(f"Error: {e}")