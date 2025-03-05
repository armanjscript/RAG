import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding

context_window = 4096

# Set the number of output tokens
num_output = 200

# Configure the LLM with the desired number of output tokens
Settings.llm = Ollama(model="qwen2.5:latest", temperature=0.5, context_window=context_window, num_output=num_output)

# Set the embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")

# Load documents from the directory
documents = SimpleDirectoryReader("data").load_data()

print(f"Number of documents: {len(documents)}")
print("================================================================================")

# Initialize ChromaDB client and collection
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")

# Set up the vector store and storage context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the vector store index from the documents
index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context
)

# Create a query engine and query the index
query_engine = index.as_query_engine()
response = query_engine.query("What are The Potential Benefits of Social Media Use Among Children and Adolescents?")
print(response)