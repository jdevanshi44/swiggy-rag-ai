import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(page_title="Swiggy AI Assistant", layout="wide")
st.title("🤖 Swiggy AI Assistant")

# --- Helper Function: Format Documents ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Core RAG Logic ---
def setup_rag(file_path):
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    # 2. Split Text (Optimized for financial reports)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(data)
    
    # 3. Embeddings & Vector Store
    # Note: Ensure sentence-transformers is in requirements.txt
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 4. Initialize LLM via Groq (Uses Streamlit Secrets)
    # This line will only work if GROQ_API_KEY is in your Secrets
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=st.secrets["GROQ_API_KEY"], 
        model_name="llama-3.3-70b-versatile"
    )
    
    # 5. Prompt Design
    template = """You are a helpful AI assistant for Swiggy. 
    Use the following pieces of retrieved context to answer the question. 
    If the answer isn't in the context, say you don't know. 
    Do not use outside information.

    Context:
    {context}

    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 6. Build the Chain (LCEL)
    # We define rag_chain explicitly here to avoid NameError
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# --- Sidebar UI ---
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload Swiggy Report (PDF)", type="pdf")
    process_btn = st.button("Initialize Analyst")

# --- Main Execution ---
if process_btn:
    if uploaded_file is not None:
        with st.spinner("Analyzing document..."):
            # Save the uploaded file locally for the loader
            with open("temp_report.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Call the function (No key passed, it's inside st.secrets)
            chain, retriever_obj = setup_rag("temp_report.pdf")
            
            # Store in session state so it persists across refreshes
            st.session_state.rag_chain = chain
            st.session_state.retriever = retriever_obj
            st.success("Analysis Complete! Ask your questions below.")
    else:
        st.warning("Please upload a PDF file first.")

# --- Chat Interface ---
query = st.text_input("What would you like to know from the Swiggy report?")

if query:
    if "rag_chain" in st.session_state:
        with st.spinner("Searching for answers..."):
            try:
                response = st.session_state.rag_chain.invoke(query)
                st.subheader("Answer:")
                st.write(response)
                
                # Show sources (for professionalism)
                with st.expander("View Source Evidence"):
                    docs = st.session_state.retriever.invoke(query)
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                        st.info(doc.page_content)
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
    else:
        st.info("Please initialize the analyst from the sidebar to begin.")