import streamlit as st
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Swiggy AI Assistant Pro", layout="wide")
st.title("🤖 Swiggy AI Assistant")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag(file_path):
    # 1. Advanced Loader: PyMuPDF is much better at preserving table structures
    # and can extract text from images if rapidocr is installed
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    
    # 2. Strategic Chunking
    # Smaller chunks with significant overlap ensure that "minute details" 
    # and numbers in tables aren't cut off from their headers.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=120,
        separators=["\n\n", "\n", "|", ".", " "] # Added pipe symbol for tables
    )
    chunks = splitter.split_documents(data)
    
    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Vector Store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 5. Increase Retrieval Count (k=7)
    # Annual reports are dense; we need more context for "minute details"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    
    # 6. LLM (Using Llama 3.3 70B for its high reasoning capabilities)
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=st.secrets["GROQ_API_KEY"], 
        model_name="llama-3.3-70b-versatile"
    )
    
    # 7. Financial Analyst Prompt
    template = """You are an expert Financial Analyst. Answer the question based ONLY on the provided context.
    The context may contain tables where data is separated by spaces or bars. 
    Pay close attention to numbers, dates, and currency (INR/Millions).
    
    If the answer involves a list or a table, format it clearly using bullet points or markdown tables.
    If the context doesn't contain the answer, say "This specific detail is not mentioned in the report."

    Context:
    {context}

    Question: {question}
    
    Detailed Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# --- UI Logic ---
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload Swiggy Report", type="pdf")
    if st.button("Deep Sync Document"):
        if uploaded_file:
            with st.spinner("Processing text and tables..."):
                with open("temp_report.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                chain, retriever_obj = setup_rag("temp_report.pdf")
                st.session_state.rag_chain = chain
                st.session_state.retriever = retriever_obj
                st.success("Deep Analysis Complete!")

query = st.text_input("Ask a highly specific question (e.g., specific expenses or page-level stats):")

if query and "rag_chain" in st.session_state:
    with st.spinner("Scanning document for details..."):
        response = st.session_state.rag_chain.invoke(query)
        st.markdown("### Analysis Result:")
        st.write(response)
        
        with st.expander("View Source Context (Verification)"):
            docs = st.session_state.retriever.invoke(query)
            for doc in docs:
                st.write(f"**Page {doc.metadata.get('page')+1}:**")
                st.caption(doc.page_content)