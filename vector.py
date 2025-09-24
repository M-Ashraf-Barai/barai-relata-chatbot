from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Debug info
print("Current Working Directory:", os.getcwd())
print("File Exists:", os.path.exists("my_family.csv"))

# ✅ Load the biography file as raw text
if not os.path.exists("my_family.csv"):
    raise FileNotFoundError("CSV file not found.")

with open("my_family.csv", "r", encoding="utf-8") as f:
    biography_text = f.read()

# ✅ Split into chunks for better embedding performance
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # you can adjust
    chunk_overlap=50
)
chunks = text_splitter.split_text(biography_text)

# ✅ Wrap each chunk as a Document with metadata
documents = [
    Document(
        page_content=chunk,
        metadata={"source": "my_family.csv", "type": "biography"}
    )
    for chunk in chunks
]

# ✅ Set up embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# ✅ Build or load FAISS
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

# # ✅ Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
