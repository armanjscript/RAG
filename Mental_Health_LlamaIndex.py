from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.ollama import OllamaEmbedding
import os

# Settings.llm = Ollama(model="mistral:latest", temperature=0.2)
Settings.llm = Ollama(model="qwen2.5:latest", temperature=0.2)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")

os.environ["OPEN_API_KEY"] = ""

documents = SimpleDirectoryReader("data").load_data() 
index = VectorStoreIndex.from_documents(
    documents
)

# print(f"Number of documents: {len(documents)}")
# print(f"Display a document in the Index: {documents[25].text}")
# print("=============================================")


retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10
)

response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(
        similarity_cutoff=0.7,
        filter_empty=True,
        filter_duplicates=True,
        filter_similar=True
    )],
    response_synthesizer=response_synthesizer
)

response = query_engine.query("What are The Potential Benefits of Social Media Use Among Children and Adolescents?")
print(response)