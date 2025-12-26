# Annual Report Analyst (RAG Project)

This project is an **AI-powered Annual Report Analyst** built to analyze **Indian company annual reports / integrated reports** using a **Retrieval-Augmented Generation (RAG)** approach.

It allows users to upload a PDF annual report and ask questions related to **financials, risks, compliance, cash flow, and management discussion**, with answers grounded in the document.

---

## What This Project Does

* Upload an Indian annual report (PDF)
* Index the document using embeddings
* Retrieve only relevant sections (financials, risks, notes, MD&A)
* Answer user questions using a Large Language Model
* Show source pages and confidence for answers

This project is designed for **portfolio and interview demonstration**, focusing on real-world document analysis.

---

## Key Features

* RAG-based document question answering
* History-aware retrieval for follow-up questions
* Section-aware filtering (ignores AGM and voting procedures)
* Source citations with page numbers
* Streaming responses for better user experience

---

## Example Questions You Can Ask

* What is the standalone vs consolidated revenue?
* What are the major risk factors mentioned?
* Are there any red flags in cash flow?
* What dividend has been declared?
* Summarize management discussion and outlook
* Any related party transactions disclosed?

---

## Tech Stack

* Python
* Streamlit (UI)
* LangChain (RAG framework)
* Chroma (vector database)
* HuggingFace embeddings (BGE / E5)
* Groq LLM (low-latency inference)
* PyMuPDF (PDF parsing)

---

## How It Works (High Level)

1. User uploads an annual report PDF
2. The document is cleaned and irrelevant sections are removed
3. Text is split into chunks and embedded
4. Relevant chunks are retrieved based on the question
5. The LLM generates an answer using retrieved context
6. Sources and confidence are shown to the user

---

## How to Run the Project

```bash
git clone https://github.com/gordhan04/RAG_AnnualReportAnalyse.git
cd annual-report-analyst

pip install -r requirements.txt

# Add API key in .env file
streamlit run app.py
```

---

## Why This Project Is Useful

* Demonstrates real-world RAG implementation
* Handles long, compliance-heavy Indian reports
* Shows understanding of financial documents
* Designed with performance and retrieval quality in mind

This project is suitable for roles related to:

* Data Science
* AI / ML Engineering
* Applied NLP
* RAG-based systems

---

## Author

Govardhan Purohit
