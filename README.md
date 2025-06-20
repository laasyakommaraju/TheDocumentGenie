# The Document Genie 🧞

This web application transforms how users interact with their documents. The Document Genie allows you to upload a PDF and ask it questions in natural, conversational language. It leverages the power of Large Language Models (LLMs) to provide intelligent, context-aware answers sourced directly from the document's content.

This project streamlines information retrieval, enabling users to instantly find the knowledge they need without manually searching through pages of text.

### ✨ Features

* **Intelligent Q&A**: Ask complex questions and receive accurate, context-aware answers.
* **PDF Document Support**: Upload any text-based PDF to begin a conversation.
* **Conversational Interface**: A simple and intuitive UI, built with Streamlit, for a seamless user experience.
* **High-Performance AI**: Utilizes Google's Gemini models for state-of-the-art text generation and understanding.
* **Secure & Self-Contained**: Runs locally and securely manages your API keys using environment variables.

### 🛠️ Technology Stack

* **Language**: Python
* **Web Framework**: Streamlit
* **AI Orchestration**: LangChain & LangChain Expression Language (LCEL)
* **LLM & Embeddings**: Google Gemini & `embedding-001`
* **Vector Database**: FAISS (Facebook AI Similarity Search)
* **PDF Processing**: PyPDF2

### 🚀 Installation

Follow these steps to set up and run the project on your local machine.

**Prerequisites**

* Python 3.8+
* pip (Python package installer)

**Steps**

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/document-genie.git](https://github.com/your-username/document-genie.git)
    cd document-genie
    ```

2.  **Install Dependencies**
    Create a `requirements.txt` file with the following content:
    ```
    streamlit
    python-dotenv
    PyPDF2
    langchain
    langchain-google-genai
    faiss-cpu
    ```
    Then, run the installation command:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Environment Variables**
    Create a `.env` file in the root directory of the project:
    ```bash
    touch .env
    ```
    Edit the `.env` file and add your Google API Key:
    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

4.  **Run the Application**
    Start the Streamlit server:
    ```bash
    streamlit run main.py
    ```

5.  **Access the Application**
    Open your browser and go to `http://localhost:8501`.

### 🏛️ Architecture: How It Works

The Document Genie is built on a **Retrieval-Augmented Generation (RAG)** pipeline, which ensures answers are grounded in the content of the provided document.

1.  **Ingestion & Extraction**: The application accepts a PDF file and uses `PyPDF2` to extract all text content.
2.  **Chunking**: The extracted text is broken down into smaller, overlapping chunks using `LangChain`'s `CharacterTextSplitter`.
3.  **Embedding**: Each chunk is converted into a numerical representation (a vector) using Google's embedding model. These vectors capture the semantic meaning of the text.
4.  **Indexing**: The vectorized chunks are stored in a `FAISS` vector store, creating a searchable "mind palace" of the document's knowledge.
5.  **Retrieval & Generation**: When a user asks a question, it is also converted into a vector. The system searches the FAISS store to find the most relevant chunks from the document. These chunks, along with the original question, are passed to the **Google Gemini model**, which generates a final, coherent answer.


### 🤝 Contributing

Contributions are welcome! Please feel free to open an issue to discuss a new feature or submit a pull request for any improvements or bug fixes.
