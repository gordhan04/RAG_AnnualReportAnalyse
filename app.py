import streamlit as st
import os
import tempfile
from rag_engine import process_document_to_chroma, get_rag_chain, format_docs
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage # <--- Import for History

load_dotenv()

st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #0e1117;
    color: #eaeaea;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
}

/* Chat bubbles */
div[data-testid="stChatMessage"] {
    background-color: #161b22;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
}

/* User message */
div[data-testid="stChatMessage"]:has(span:contains("user")) {
    background-color: #1f6feb;
}

/* Input box */
textarea {
    border-radius: 10px !important;
}

/* Buttons */
button {
    border-radius: 10px !important;
    background-color: #238636 !important;
    color: white !important;
}

/* File uploader */
section[data-testid="stFileUploader"] {
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Remove Streamlit default top padding */
.block-container {
    padding-top: 1.8rem !important;
}

/* Header box spacing */
.header-box {
    margin-bottom: 80px; /* space before "Try asking" */
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box" style="
background: linear-gradient(90deg, #1f6feb, #238636);
padding: 20px;
border-radius: 16px;
color: white;">
<h2 style="margin: 0;">📊 Annual Report Analyst</h2>
<p style="margin-top: 8px;">AI-powered RAG system for financial document analyzing</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 💡 Try asking:")
st.markdown("""
- What is the company’s revenue growth over last 3 years?  
- What are the major risk factors mentioned?  
- Summarize management strategy  
- Any red flags in cash flow?
""")

    
st.set_page_config(page_title="Annual Report Analyst", layout="wide")
# st.title("📊 Annual Report Analyst (with Memory)")

# Initialize Chat History in Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Upload Report")
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Supports financial statements, risk sections, notes"
    )

    if uploaded_file and "vectorstore" not in st.session_state:
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            st.session_state.vectorstore = process_document_to_chroma(tmp_path)
            st.success("Report Indexed!")
            os.remove(tmp_path)
    st.markdown("---")
    st.markdown("### ⚙️ Capabilities")
    st.markdown("""
    - Revenue Analysis  
    - Risk Factors  
    - Management Discussion  
    - Strategy Insights  
    """)

# Display Messages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

# ... (Previous code for sidebar and history display remains the same) ...

if prompt := st.chat_input("Ask about the report..."):
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    if "vectorstore" in st.session_state:
        # Get chains
        retriever_chain, generation_chain = get_rag_chain(st.session_state.vectorstore)
        
        # Start the Assistant Response
        with st.chat_message("assistant"):
            # A. RETRIEVAL STEP (Cannot stream this, but it's fast)
            with st.spinner("Searching documents..."):
                retrieved_docs = retriever_chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": prompt
                })
            
            # B. FORMATTING
            formatted_context = format_docs(retrieved_docs)

            # C. GENERATION STEP (Streaming)
            # We create a generator object using .stream() instead of .invoke()
            stream_generator = generation_chain.stream({
                "context": formatted_context,
                "chat_history": st.session_state.chat_history,
                "input": prompt
            })
            
            # st.write_stream automatically loops through the generator and types the text!
            # It also returns the final complete string, which we need for history.
            response_text = st.write_stream(stream_generator)
            
            # D. SAVE TO HISTORY
            # We use the full 'response_text' captured above
            st.session_state.chat_history.append(AIMessage(content=response_text))
            
            # E. SHOW SOURCES (After the stream finishes)
            with st.expander("View Sources"):
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Source {i+1}**")
                    page = doc.metadata.get("page", "Unknown")
                    st.caption(f"Page: {page}")
                    st.info(doc.page_content[:300] + "...")
                    st.divider()

    else:
        st.error("Please upload a document first.")