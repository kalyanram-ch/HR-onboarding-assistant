import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_DB_DIR = "HR_Vector_DB"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vector_db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k":5}
)


OLLAMA_MODEL_NAME = "phi3"

llm = ChatOllama(model=OLLAMA_MODEL_NAME)


prompt = ChatPromptTemplate.from_template("""
You are an HR Onboarding Assistant.

Use the provided HR documents to answer employee questions
about company policies, benefits, and onboarding.

If the answer is not available in the documents, say:
"I could not find that information in the HR policy documents."

Context:
{context}

Question:
{question}
""")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

st.title("HR Onboarding Assistant")

st.write("Ask questions about HR policies, benefits, and onboarding.")

user_question = st.text_input("Enter your question")

if user_question:
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_question)
        st.write("### Answer")
        st.write(response)