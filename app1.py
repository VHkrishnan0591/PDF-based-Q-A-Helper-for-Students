import os
import streamlit as st
from dotenv import load_dotenv
from langserve import RemoteRunnable
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers.string import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.messages import trim_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import chromadb.api

global filename
global with_message_history


st.title('PDF STUDENT HELPER')
st.sidebar.title("Document Setting")
uploaded_file = st.sidebar.file_uploader("Upload a file", type=None)
# Extract the file name

st.sidebar.title("Chat Settings")

model_name = st.sidebar.selectbox(
    "Choose a llm model",
    ("mixtral-8x7b-32768", "gemma2-9b-it","llama3-8b-8192")
)
temperature = st.sidebar.slider("Give a temperature", 0.0, 1.0)

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")


def initiate_class(filename):
    loader = PyPDFLoader(filename)
    docs = loader.load()
    # Recurrsive Split of documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1500,
        chunk_overlap= 50,
    )
    documents = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name = 'ibm-granite/granite-embedding-125m-english')
    st.session_state.vector_database = Chroma.from_documents(documents=documents, embedding=embeddings,persist_directory="./chroma_db"  )
    chromadb.api.client.SharedSystemClient.clear_system_cache()

if uploaded_file is not None:
    # Extract the file name
    filename = uploaded_file.name
    # Define the save path
    save_path = f"./{uploaded_file.name}"  # Saves the file in the current directory

    # Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{uploaded_file.name}' has been saved successfully!")
def create_llm_prompt_chain():
    # LLM MOdel 
    models = ChatGroq(temperature=temperature, model_name=model_name)
    parser = StrOutputParser()
    system = "You are Robotics and Autonomous system Engineer and you should answer the question according to the {context} only."
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"),("human", "{input}")])
    st.session_state.retriever = st.session_state.vector_database.as_retriever(search_type="mmr")
    trimmer = trim_messages(
        max_tokens=5000,
        strategy="last",
        token_counter=models,
        # Usually, we want to keep the SystemMessage
        # if it's present in the original history.
        # The SystemMessage has special instructions for the model.
        include_system=True,
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        # start_on="human" makes sure we produce a valid chat history
        start_on="human",
    )
    chain = create_stuff_documents_chain(trimmer| models, prompt, output_parser=parser) 
    retrieval_chain = create_retrieval_chain(st.session_state.retriever, chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return (st.session_state.store[session_id])

    with_message_history = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )
    return with_message_history
    
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = {}
if "vector_database" not in st.session_state:
    st.session_state.vector_database = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if st.button("Start PDF Loader after uploading PDF"):
    try:
        initiate_class(filename)
        st.write("The PDF file was successfully started")
    except Exception as e:
        if str(e) == "name 'filename' is not defined":
            st.warning("Upload a PDF please")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def response_generator(input):
    with_message_history = create_llm_prompt_chain()
    response = with_message_history.invoke(
    {"input": input},
    config={"configurable": {"session_id": "abc123"}},
    )
    return response['answer']

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
            try:
                response = response_generator(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                if str(e) == "'NoneType' object has no attribute 'as_retriever'":
                    st.warning("Upload a PDF please")

    # Add assistant response to chat history
    



