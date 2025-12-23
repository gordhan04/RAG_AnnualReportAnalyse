# Streamlit frontend will be here
import streamlit as st
import os
import tempfile
from rag_engine import process_document_to_chroma, get_rag_chain
from dotenv import load_dotenv

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
<div style="
background: linear-gradient(90deg, #1f6feb, #238636);
padding: 20px;
border-radius: 16px;
color: white;">
<h2>📊 Annual Report Analyst</h2>
<p>AI-powered RAG system for financial document intelligence</p>
</div>
""", unsafe_allow_html=True)


# st.set_page_config(page_title="Company Annual Report Analyst", layout="wide")
# st.title("📊 Company Annual Report Analyst")

col1, col2 = st.columns([3, 1])

# with col1:
#     st.subheader("💬 Ask Questions")
st.markdown("### 💡 Try asking:")
st.markdown("""
- What is the company’s revenue growth over last 3 years?  
- What are the major risk factors mentioned?  
- Summarize management strategy  
- Any red flags in cash flow?
""")
    
# with col1:
#     st.subheader("📄 Report Status")
#     if "vectorstore" in st.session_state:
#         st.success("Indexed")
#     else:
#         st.warning("No report uploaded")

# Sidebar for Setup
with st.sidebar:
    st.markdown("## 📂 Document Control")
    st.caption("Upload annual reports (10-K / India Annual Reports)")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Supports financial statements, risk sections, notes"
    )

    
    if uploaded_file and "vectorstore" not in st.session_state:
        with st.spinner("Processing with Docling (Parsing Tables...)..."):
            # Save temp file for Docling to read
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Process
            st.session_state.vectorstore = process_document_to_chroma(tmp_path)
            st.success("Report Indexed Successfully!")
            os.remove(tmp_path) # Cleanup
    st.markdown("---")
    st.markdown("### ⚙️ Capabilities")
    st.markdown("""
    - Revenue Analysis  
    - Risk Factors  
    - Management Discussion  
    - Strategy Insights  
    """)

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Revenue, Risks, or Strategy..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if "vectorstore" in st.session_state:
        chain = get_rag_chain(st.session_state.vectorstore)
        with st.spinner("Analyzing..."):
            response = chain.invoke({"input": prompt})
            answer = response["answer"]
            
            # Append Sources (Portfolio "Wow" Factor)
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in response['context']]))
            # Note: Docling metadata might need inspection to get exact page numbers perfectly, 
            # but this is a good start.
            
    else:
        answer = "Please upload a document first to start the analysis."

    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown("### 🧠 Analysis Result")
        st.markdown(answer)
    