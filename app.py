import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

# Load environment variables
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up custom CSS for chat layout
st.markdown(
    """
    <style>
    .main {
        background-color: black;
        color: white;
    }
    .chat-message {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .user {
        margin-left: auto;
        background-color: #1e90ff;
        color: white;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
    }
    .assistant {
        margin-right: auto;
        background-color: #f0f0f0;
        color: black;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set up Streamlit 
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

# Function to retrieve session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state['store']:
        st.session_state['store'][session_id] = ChatMessageHistory()
    return st.session_state['store'][session_id]

# Function to display history in bot-user format with aligned chat bubbles
def display_chat_history(history):
    st.markdown("### Chat History:")
    for message in history.messages:
        if message.type == "human":
            st.markdown(
                f'<div class="chat-message"><div class="user">{message.content}</div></div>',
                unsafe_allow_html=True,
            )
        elif message.type == "ai":
            st.markdown(
                f'<div class="chat-message"><div class="assistant">{message.content}</div></div>',
                unsafe_allow_html=True,
            )

# Check if Groq API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Initialize session store if not present
    if 'store' not in st.session_state:
        st.session_state['store'] = {}

    # Chat interface with session ID
    session_id = st.text_input("Session ID", value="default_session")

    uploaded_files = st.file_uploader("Choose A PDF file", type="pdf", accept_multiple_files=True)

    # Process uploaded PDFs
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp_{uploaded_file.name}"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer question
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Get user input for the question
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            # Display the assistant's response
            st.write("Assistant:", response['answer'])

            # Display chat history in user-friendly format
            display_chat_history(session_history)
else:
    st.warning("Please enter the Groq API Key")
