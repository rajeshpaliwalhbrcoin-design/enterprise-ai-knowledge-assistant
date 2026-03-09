# Enterprise AI Knowledge Assistant

An enterprise-grade **Retrieval-Augmented Generation (RAG)** assistant that enables employees to search internal documentation using natural language queries.

This system combines **vector search, semantic embeddings, and large language models (LLMs)** to deliver contextual and accurate answers from enterprise knowledge bases.

---

## Key Features

* Semantic document search using vector embeddings
* Retrieval-Augmented Generation (RAG) architecture
* Fast similarity search using **FAISS vector database**
* LLM-powered responses using **Ollama / LLM APIs**
* Interactive **Streamlit-based chat interface**
* Semantic caching for faster repeated queries
* Document ingestion from PDFs

---

## Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **FAISS**
* **Sentence Transformers**
* **Ollama (LLM)**
* **SQLite**

---

## Use Case

This assistant is designed for **enterprise knowledge management**, allowing employees and IT teams to quickly retrieve:

* Internal policies
* Technical documentation
* Troubleshooting guides
* Knowledge base articles

Instead of manually searching documents, users can simply **ask questions in natural language**.

---

## Project Structure

```
enterprise-ai-knowledge-assistant
│
├── app.py
├── requirements.txt
├── README.md
│
├── data
│   └── (document files)
│
├── assets
│   └── (logo / UI assets)
│
└── screenshots
    └── (demo images)
```

---

## Installation

Install the required dependencies:

```
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit application:

```
streamlit run app.py
```

The assistant will launch in your browser.

---

## Future Improvements

* Multi-document ingestion pipeline
* Enterprise authentication integration
* Conversation memory and analytics
* Deployment using Docker / cloud infrastructure

---

![AI Assistant Demo] - screenshots/demo.png

## Architecture

User Question
↓
Embedding Model (Sentence Transformers)
↓
Vector Search (FAISS)
↓
Context Retrieval
↓
LLM (Ollama / Llama)
↓
Generated Response
↓
Streamlit Chat Interface


Note - Sample documents are included for demonstration purposes only.
