# import basics
import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

#documents
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv() 

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# initialize pinecone database
index_name = os.getenv("PINECONE_INDEX_NAME", "default-index")  # change if desired

# check whether index exists, and create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # bge-small-en-v1.5 produces 384-dimensional embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# initialize embeddings model + vector store
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'}
)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# loading documents from directory (supports multiple file types)
loader = DirectoryLoader(
    "data/", 
    glob="**/*",  # Load all files recursively
    show_progress=True,
    use_multithreading=True,
    # Supported file extensions will be auto-detected
    # Includes: .txt, .md, .pdf, .docx, .csv, .json, .html, etc.
)

raw_documents = loader.load()

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
documents = text_splitter.split_documents(raw_documents)

# generate unique id's

i = 0
uuids = []

while i < len(documents):

    i += 1

    uuids.append(f"id{i}")

# add to database

vector_store.add_documents(documents=documents, ids=uuids)