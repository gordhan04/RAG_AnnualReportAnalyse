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

if prompt := st.chat_input("Ask about the report..."):
    
    # 1. Display User Message Immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Add to History (as HumanMessage)
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    if "vectorstore" in st.session_state:
        retriever_chain, generation_chain = get_rag_chain(st.session_state.vectorstore)
        
        with st.spinner("Thinking..."):
            # STEP 1: Retrieve (History Aware)
            # We pass 'chat_history' so it can rewrite the query if needed
            retrieved_docs = retriever_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "input": prompt
            })
            
            # Formatting docs for the generation step
            formatted_context = format_docs(retrieved_docs)

            # STEP 2: Generate Answer
            answer = generation_chain.invoke({
                "context": formatted_context,
                "chat_history": st.session_state.chat_history,
                "input": prompt
            })
            
            # Display Answer
            with st.chat_message("assistant"):
                st.markdown(answer)
                
            # Add to History (as AIMessage)
            st.session_state.chat_history.append(AIMessage(content=answer))
            

            # Optional: Show Specific Metadata Only
            with st.expander("View Sources"):
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"### Source {i+1}:")
                    
                    # 1. Extract Specific Metadata Fields safely
                    # .get("key", "default value") prevents errors if the key is missing
                    page_num = doc.metadata.get("page", "Unknown")
                    header_info = doc.metadata.get("Header 2", "N/A") # Only works if using Markdown Splitter
                    source_file = doc.metadata.get("file_path", "Uploaded Document")
                    
                    # 2. Display them in a clean row
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**📄 Page:** {page_num}")
                    with col2:
                        st.markdown(f"**📑 Section:** {header_info}")
                    with col3:
                        # Only show filename, not the full temp path
                        clean_name = os.path.basename(source_file)
                        st.markdown(f"**📁 File:** {clean_name}")
                    
                    # 3. Show the Text Snippet
                    st.markdown("**Content:**")
                    st.info(doc.page_content[:400] + "...") # Use .info for a nice blue box look
                    
                    st.divider()
    else:
        st.error("Please upload a document first.")