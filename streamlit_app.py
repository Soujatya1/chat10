import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import os
import pandas as pd

st.title("Document GeN-ie")
st.subheader("Chat with your documents")

# Create directory if it doesn't exist
pdfs_directory = '.github/'
os.makedirs(pdfs_directory, exist_ok=True)

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. As per the question asked, please mention the accurate and precise related information. Use point-wise format, if required.
Also answer situation-based questions derived from the context as per the question.

The document may contain tables. Tables are formatted as CSV data and preceded by [TABLE] markers.

Question: {question} 
Context: {context} 
Answer:
"""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)
model = ChatGroq(groq_api_key="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", model_name="llama-3.3-70b-versatile", temperature=0.3)

def upload_pdf(file):
    file_path = pdfs_directory + file.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def extract_tables_from_pdf(file_path):
    import pdfplumber
    
    # Initialize a list to store all text and tables
    document_content = []
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text from the page
            text = page.extract_text() or ""
            if text.strip():
                document_content.append({
                    "content": text,
                    "page": page_num + 1,
                    "type": "text"
                })
            
            # Extract tables from the page
            tables = page.extract_tables()
            for table_num, table in enumerate(tables):
                if table:
                    # Convert table to DataFrame for easier manipulation
                    df = pd.DataFrame(table)
                    
                    # Clean headers (first row might be headers)
                    if not df.empty:
                        # Use the first row as headers if it exists
                        if not df.iloc[0].isna().all():
                            headers = df.iloc[0].tolist()
                            df = df.iloc[1:]
                            df.columns = headers
                    
                    # Convert to CSV string
                    csv_string = df.to_csv(index=False)
                    
                    # Add the table as a special content type
                    table_content = f"[TABLE] Table {table_num+1} on Page {page_num+1}:\n{csv_string}"
                    document_content.append({
                        "content": table_content,
                        "page": page_num + 1,
                        "type": "table"
                    })
    
    return document_content

def load_pdf(file_path):
    # Custom loading using the table extraction function
    document_content = extract_tables_from_pdf(file_path)
    
    # Convert to LangChain document format
    from langchain_core.documents import Document
    documents = []
    
    for item in document_content:
        # Create a document with the content and metadata
        doc = Document(
            page_content=item["content"],
            metadata={
                "page": item["page"],
                "source": file_path,
                "type": item["type"]
            }
        )
        documents.append(doc)
    
    return documents

def split_text(documents):
    # Use a splitter that respects table boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Process text and table documents differently
    split_docs = []
    for doc in documents:
        # Keep tables intact (don't split)
        if doc.metadata["type"] == "table":
            split_docs.append(doc)
        else:
            # Split text documents
            split_docs.extend(text_splitter.split_documents([doc]))
    
    return split_docs

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    response = chain.invoke({"question": question, "context": context})
    
    return response.content

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_file:
    all_documents = []
    for file in uploaded_file:
        file_path = upload_pdf(file)
        documents = load_pdf(file_path)
        chunked_documents = split_text(documents)
        all_documents.extend(chunked_documents)
    
    index_docs(all_documents)
    
    # Display a success message
    st.success(f"Successfully processed {len(uploaded_file)} PDF(s) with text and tables.")
    
    # Display table preview (optional)
    table_docs = [doc for doc in all_documents if doc.metadata["type"] == "table"]
    if table_docs:
        with st.expander("Preview Extracted Tables"):
            for i, doc in enumerate(table_docs[:3]):  # Show first 3 tables only
                st.write(f"**{doc.metadata['source']} (Page {doc.metadata['page']})**")
                table_content = doc.page_content.replace("[TABLE] ", "")
                st.text(table_content)
                if i < len(table_docs[:3]) - 1:
                    st.divider()
    
    question = st.chat_input("Ask a question:")
    if question:
        st.session_state.conversation_history.append({"role": "user", "content": question})
        
        related_documents = retrieve_docs(question)
        
        answer = answer_question(question, related_documents)
        
        st.session_state.conversation_history.append({"role": "assistant", "content": answer})

for message in st.session_state.conversation_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])
