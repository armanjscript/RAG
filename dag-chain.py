from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
import os

os.environ["COHERE_API_KEY"] = "YOUR_COHERE_API_KEY"

Settings.llm = Ollama(model="qwen2.5:latest")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents=documents
)

print(f"Number of documents: {len(documents)}")

prompt_str = "What are the potential Risks of Harm from {topic}?"
prompt_tmpl = PromptTemplate(prompt_str)


retriever = index.as_retriever(similarity_top_k=5)

reranker = CohereRerank(api_key=os.getenv("COHERE_API_KEY"))

summarizer = TreeSummarize()

p = QueryPipeline(verbose=True)

p.add_modules({
    "llm": Settings.llm,
    "prompt_tmpl": prompt_tmpl,
    "retriever": retriever,
    "summarizer": summarizer,
    "reranker": reranker
})

p.add_link("prompt_tmpl", "llm")
p.add_link("llm", "retriever")
p.add_link("retriever", "reranker", dest_key="nodes")
p.add_link("llm", "reranker", dest_key="query_str")
p.add_link("reranker", "summarizer", dest_key="nodes")
p.add_link("llm", "summarizer", dest_key="query_str")

output = p.run(topic="Content Exposure")
print(output)