import os
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama # For connecting to local Ollama models
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

# --- Configuration ---
HR_VECTOR_DB_DIR = "HR_Vector_DB"
HR_DOCS_SOURCE_DIR = "HR_Documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "phi3" # Ensure this matches the model you pulled in Ollama (e.g., llama3.1, llama2, mistral)

# --- Function to Load Documents (reused from previous script) ---
def load_hr_documents(source_dir: str) -> list:
    """
    Loads documents from the specified directory using appropriate LangChain loaders.
    Supports .txt, .pdf, and .docx files.
    """
    print(f"\nLoading documents from '{source_dir}'...")
    loaded_documents = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith((".txt",".py")):
                    loader = TextLoader(file_path)
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif file.endswith(".pptx"):
                    loader = UnstructuredPowerPointLoader(file_path)
                elif file.endswith(".json"):
                    from langchain_community.document_loaders import UnstructuredFileLoader
                    loader = UnstructuredFileLoader(file_path)
                elif file.endswith(".xlsx"):
                    from langchain_community.document_loaders import UnstructuredExcelLoader
                    loader = UnstructuredExcelLoader(file_path)
                else:
                    print(f"  Skipping unsupported file type: {file_path}")
                    continue
                loaded_documents.extend(loader.load())
                print(f"  Loaded: {file}")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
    if not loaded_documents:
        print(f"No documents found or loaded from '{source_dir}'.")
    return loaded_documents

# --- Function to Create or Load the Vector Database (reused from previous script) ---
def get_or_create_hr_vector_db(docs_source_dir: str, db_persist_dir: str) -> Chroma:
    """
    Creates or loads the Chroma vector database for the HR Team.
    It will load documents, chunk them, embed them, and store/load them.
    """
    print(f"\n--- Setting up HR Knowledge Base ---")
    os.makedirs(docs_source_dir, exist_ok=True)
    os.makedirs(db_persist_dir, exist_ok=True)

    print(f"1. Initializing embedding model: {EMBEDDING_MODEL_NAME} (This may download the model)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model initialized.")

    vectorstore = None
    if os.path.exists(os.path.join(db_persist_dir, "chroma.sqlite3")):
        print(f"2. Loading existing vector database from '{db_persist_dir}'...")
        vectorstore = Chroma(persist_directory=db_persist_dir, embedding_function=embeddings)
        print("Vector database loaded.")
    else:
        print(f"2. Creating new vector database in '{db_persist_dir}'...")

    documents = load_hr_documents(docs_source_dir)
    if not documents:
        if vectorstore:
            print("No new documents found to add. Using existing database.")
            return vectorstore
        else:
            print("Cannot create database: No documents found to process. Please add files to 'CoE_AI_Team_Work_Docs'.")
            exit()

    print("\n3. Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    if vectorstore is None:
        print(f"\n4. Populating new vector database with {len(chunks)} chunks...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_persist_dir
        )
        print("New vector database created and populated.")
    else:
        print(f"\n4. Adding {len(chunks)} new chunks to existing vector database...")
        vectorstore.add_documents(chunks)
        print("Existing vector database updated with new chunks.")
    
    vectorstore.persist()
    print(f"Vector database saved to disk in '{db_persist_dir}'.")

    return vectorstore

# --- Main Chatbot Logic ---
if __name__ == "__main__":
    # --- Ensure Ollama Server is Running ---
    print("--- IMPORTANT: Ensure your Ollama server is running and 'llama3.1' model is pulled ---")
    print("   You can run 'ollama run llama3.1' in your terminal to start it.")
    print("   If you have a different model, update OLLAMA_MODEL_NAME in the script.")
    


    # --- 1. Create or Load the CoE AI Vector Database ---
    # This will ensure your knowledge base is ready
    hr_vector_db = get_or_create_hr_vector_db(HR_DOCS_SOURCE_DIR, HR_VECTOR_DB_DIR)

    # --- 2. Initialize Ollama LLM ---
    print(f"\n2. Initializing Ollama LLM with model: {OLLAMA_MODEL_NAME}...")
    # Ensure Ollama server is running and the model is pulled
    llm = ChatOllama(model=OLLAMA_MODEL_NAME)
    print("Ollama LLM initialized.")

    # --- 3. Create a Retriever from the Vector Database ---
    print("\n3. Creating retriever from vector database...")
    retriever = hr_vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 12}
)# Retrieve top 3 relevant chunks
    print("Retriever created.")

    # --- 4. Define the RAG Prompt Template ---
    # This template instructs the LLM to use the provided context
    rag_prompt_template = ChatPromptTemplate.from_template("""
You are an HR Onboarding Assistant.

Your job is to help employees understand company policies,
benefits, leave rules, and onboarding procedures.

If the answer is present in the context, explain it clearly.
Answer only from the provided HR documents.

If the answer is not found in the context, say:
"I could not find that information in the HR policy documents."

    Context: {context}

    Question: {question}
    """)
    print("\n4. RAG Prompt Template defined.")

    # --- 5. Build the RAG Chain ---
    # This uses LangChain Expression Language (LCEL) to define the RAG workflow
    print("\n5. Building the RAG chain...")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} # Pass question to retriever, and question itself
        | rag_prompt_template                                    # Format into the RAG prompt
        | llm                                                    # Send to Ollama LLM
        | StrOutputParser()                                      # Parse LLM's output as a string
    )
    print("RAG chain built.")

    # --- 6. Start the Chatbot Loop ---
    print("\n--- HR Onboarding Assistant (Powered by Ollama & Local Vector DB) ---")
    print("Ask questions about HR policies, benefits, and onboarding. Type 'exit' to quit.")

    while True:
        user_input = input("\nYour Question: ")
        if user_input.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break
        
        print("Thinking...")
        try:
            # Invoke the RAG chain with the user's question
            response = rag_chain.invoke(user_input)
            print(f"AI Assistant: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your Ollama server is running and the model is available.")

