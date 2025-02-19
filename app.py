import streamlit as st
import os
from dotenv import load_dotenv
from langserve import RemoteRunnable
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers.string import StrOutputParser

st.title('Langserver Application')
uploaded_file = st.file_uploader("Upload a file", type=None)  # `type=None` allows all file types


# Check if a file is uploaded
if uploaded_file is not None:
    # Extract the file name
    filename = uploaded_file.name
    print(filename)
    # Define the save path
    save_path = f"./{uploaded_file.name}"  # Saves the file in the current directory

    # Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{uploaded_file.name}' has been saved successfully!")



# Load environment variables from .env file
load_dotenv()

# Access the API key
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# get the input question
topic = st.text_input('topic')

if st.button('click'):

# Initate the PDF loader
    loader = PyPDFLoader(filename)
    pages = loader.load()

    # Splitting the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=100,
        chunk_overlap=20)
    texts =[]
    for page in pages:
        texts += text_splitter.create_documents([page.page_content])

    # Sentence Embedding and storing FAISS Database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS

    index = faiss.IndexFlatL2(len(embeddings.embed_query(texts[0].page_content)))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(texts)

    # Setup the Retriever using FAISS Database
    retriever = vector_store.as_retriever(
        search_type="similarity"
        # search_kwargs={'score_threshold': 0.8}
    )
    # Setup a model
    chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

    parser = StrOutputParser()
    system_template = "You are Expert Robotics and Autonomous system Engineer and you should answer the question according to the {context}."
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{input}")]
    )
    combine_docs_chain = create_stuff_documents_chain(chat, prompt, output_parser=parser)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    response = retrieval_chain.invoke({"input": topic})
    st.write(f"The response is {response['answer']}")


