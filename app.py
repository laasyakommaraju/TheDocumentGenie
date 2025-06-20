import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    # Load all the environment variables from the .env file
    load_dotenv()

    st.set_page_config(page_title="Ask your PDF")
    
    # --- Custom CSS for Styling ---
    st.markdown("""
    <style>
    /* Main app container */
    .stApp {
        background-color: #e6f0ff; /* Light Blue */
        background-image: linear-gradient(to bottom right, #e6f0ff, #b3d1ff);
    }

    /* Main header */
    h1 {
        color: #b30000; /* Darker Red */
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }

    /* General text color */
    .st-emotion-cache-1r4qj8v, .st-emotion-cache-1y4p8pa {
        color: #cc0000; /* Red */
    }

    /* User input label */
    .st-emotion-cache-ue6h4q {
        color: #b30000; /* Darker Red */
    }
    
    /* User input box */
    .st-emotion-cache-1p5k0mc {
        border-color: #cc0000;
        color: #cc0000;
    }
    
    /* Response text from the LLM */
    .st-emotion-cache-1629p8f p {
        color: #ff3333; /* Brighter Red */
        font-family: 'Georgia', serif;
        font-size: 1.1em;
        border: 1px dashed #cc0000;
        padding: 15px;
        border-radius: 10px;
        background-color: #fff0f0; /* Very light red/pink background */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    st.header("The Document Genie 🧞")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            
            if not text.strip():
                st.warning("Could not extract any text from the PDF. The file might be image-based or corrupted.")
                st.stop()
                
            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Get the key from the environment variables loaded from your .env file
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                st.error("GOOGLE_API_KEY not found in .env file. Please add it.")
                st.stop()
            
            # create embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=api_key
            )

            # create vector store
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            
            # show user input
            user_question = st.text_input("Ask a question about your PDF:")
            if user_question:
                
                retriever = knowledge_base.as_retriever()
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    google_api_key=api_key,
                    temperature=0.7
                )

                template = """
                You are a helpful assistant. Answer the user's question based only on the context provided.
                If you don't know the answer, just say that you don't know.

                Context: {context}
                Question: {question}
                """
                prompt = ChatPromptTemplate.from_template(template)

                # This is the modern chain that does not use 'load_qa_chain'
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # Get and display the response
                with st.spinner("The Genie is thinking..."):
                    response = chain.invoke(user_question)
                st.write(response)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

