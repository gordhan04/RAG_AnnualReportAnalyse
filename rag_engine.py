import os
import shutil
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever # <--- NEW IMPORT

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

def process_document_to_chroma(uploaded_file_path):
    # Note: PyMuPDF is fast but Docling is better for Tables.
    # Proceeding with PyMuPDF as requested.
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    loader = PyMuPDFLoader(uploaded_file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        streaming=True
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # --- STEP 1: Contextualize Question (The "Memory" Part) ---
    # This prompt helps the AI understand "What is the revenue?" -> "What is the revenue of Apple?"
    # based on previous chat history.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"), # <--- History goes here
        ("human", "{input}"),
    ])
    
    # This special retriever will now run the "Reformulate" step automatically
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # --- STEP 2: Answer the Question ---
    system_prompt = (
        "You are an expert Financial Analyst reading an Annual Report. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the context contains a table, analyze the rows and columns carefully. "
        "If you don't know the answer, say that you don't know. "
        "\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"), # <--- History goes here too
        ("human", "{input}"),
    ])

    # We keep your manual LCEL pipe
    generation_chain = qa_prompt | llm | StrOutputParser()
    
    return history_aware_retriever, generation_chain