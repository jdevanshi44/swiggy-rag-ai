# Swiggy Annual Report RAG System

## Objective
[cite_start]An AI-powered system designed to accurately answer questions based strictly on the Swiggy Annual Report FY 2023-24[cite: 2, 3, 4].

## Features
- [cite_start]**Document Processing**: Automated PDF loading and chunking[cite: 13, 15].
- [cite_start]**Vector Search**: Uses FAISS and Sentence-Transformers for semantic retrieval[cite: 19].
- [cite_start]**No Hallucination**: Strictly grounded answers using Retrieval-Augmented Generation[cite: 11, 23].
- [cite_start]**Source Attribution**: Shows exactly which page the answer came from[cite: 29].

## Tech Stack
- **Framework**: LangChain
- **LLM**: Groq
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector DB**: FAISS
- **UI**: Streamlit

## Setup Instructions
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Data Source
- [cite_start]**Document**: Swiggy Annual Report FY 2023-24 [cite: 6]
- [cite_start]**Source Link**: [Swiggy Investor Relations](https://www.swiggy.com/investor-relations) [cite: 7, 8]