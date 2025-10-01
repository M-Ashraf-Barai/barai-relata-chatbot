
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found. Please set it in your .env file.")
# Debug info
print("Current Working Directory:", os.getcwd())
print("File Exists:", os.path.exists("my_family.csv"))

# Load the biography file as raw text
if not os.path.exists("my_family.csv"):
    raise FileNotFoundError("CSV file not found.")

with open("my_family.csv", "r", encoding="utf-8") as f:
    biography_text = f.read()

# Split into chunks for better embedding performance
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # you can adjust
    chunk_overlap=100
)
chunks = text_splitter.split_text(biography_text)

# Wrap each chunk as a Document with metadata
documents = [
    Document(
        page_content=chunk,
        metadata={"source": "my_family.csv", "type": "biography"}
    )
    for chunk in chunks
]


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",api_key=api_key)

# Build or load FAISS
index_path = "faiss_index"
print("Index path:", index_path)

if os.path.exists(index_path):
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    print("Loaded existing FAISS index.")
else:
    print("Generating embeddings and building FAISS index...")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(index_path)
    print("FAISS index saved.")

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
