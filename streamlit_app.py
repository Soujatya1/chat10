import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

st.title("Document GeN-ie")
st.subheader("Chat with your documents")

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. As per the question asked, please mention the accurate and precise related information. Use point-wise format, if required .
Also answer situation-based questions derived from the context as per the question.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = '.github/'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)

model = ChatGroq(groq_api_key="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", model_name="llama-3.3-70b-versatile", temperature=0)

def upload_pdf(file):
    file_path = pdfs_directory + file.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

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

    for uploaded_file in uploaded_file:
        file_path = upload_pdf(uploaded_file)
        documents = load_pdf(file_path)
        chunked_documents = split_text(documents)
        all_documents.extend(chunked_documents)
    index_docs(all_documents)

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
