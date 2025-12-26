import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
# Import your custom modules
from rag_engine import process_document_to_chroma, get_rag_chain, format_docs
from ui_components import apply_custom_styles, render_header, render_sidebar_capabilities

load_dotenv()
# Initialize a session ID to track resets
if "id" not in st.session_state:
    st.session_state.id = 0

def reset_application():
    """Wipes the DB data and resets the session."""
    
    # 1. Logical Wipe 
    if "vectorstore" in st.session_state:
        try:
            # Delete all documents in the collection
            st.session_state.vectorstore.delete_collection()
            st.toast("Database cleared successfully")

        except Exception as e:
            st.toast(f"⚠️ Error clearing collection: {e}")
        
        # NOW we can remove the reference from memory
        del st.session_state.vectorstore 
    
    # 2. Physical Wipe (Optional Backup)
    # We try to delete the folder, but if Windows blocks it, we ignore it.
    # Since we already emptied the DB in Step 1, it doesn't matter if the folder stays!
    # if os.path.exists("./chroma_db"):
    #     try:
    #         shutil.rmtree("./chroma_db")
    #         print("✅ Folder deleted.")
    #     except PermissionError:
    #         print("🔒 Windows locked the folder. Keeping empty folder (safe).")
    #     except Exception as e:
    #         print(f"⚠️ Other deletion error: {e}")

    # 3. Clear History
    st.session_state.chat_history = []
    
    # 4. Force UI Reset
    st.session_state.id += 1 
    
    st.rerun()

# 1. Page Config (Must be the first Streamlit command)
st.set_page_config(page_title="Annual Report Analyst", layout="wide")

# 2. Apply UI Components
apply_custom_styles()
render_header()

# 3. Session State Init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 4. Sidebar Logic
with st.sidebar:
    st.header("Upload Report")
    
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key=f"uploader_{st.session_state.id}",
        help="Supports financial statements, risk sections, notes"
    )

    if st.button("🔄 Reset App", type="primary"):
        reset_application()
        
    if uploaded_file and "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Process using your backend engine
            st.session_state.vectorstore = process_document_to_chroma(tmp_path)
            st.success("Report Indexed Successfully!")
            os.remove(tmp_path)
    
    # Render static sidebar text
    render_sidebar_capabilities()

# 5. Render Existing Chat History
for message in st.session_state.chat_history:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(message.content)

# 6. Main Chat Loop
if prompt := st.chat_input("Ask about the report..."):
    
    # Display & Save User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    if "vectorstore" in st.session_state:
        # Get the chain
        retriever_chain, generation_chain = get_rag_chain(st.session_state.vectorstore)
        
        with st.chat_message("assistant"):
            # A. Retrieval
            with st.spinner("Analyzing document..."):
                retrieved_docs = retriever_chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": prompt
                })
            
            # B. Generation (Streaming)
            formatted_context = format_docs(retrieved_docs)
            
            stream_generator = generation_chain.stream({
                "context": formatted_context,
                "chat_history": st.session_state.chat_history,
                "input": prompt
            })
            
            response_text = st.write_stream(stream_generator)
            st.session_state.chat_history.append(AIMessage(content=response_text))
            
            # C. Sources
            with st.expander("View Sources"):
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Source {i+1}**")
                    page = doc.metadata.get("page", "Unknown")
                    st.caption(f"Page: {page}")
                    st.info(doc.page_content[:300] + "...")
                    st.divider()
    else:
        st.error("Please upload a document first to start the analysis.")

