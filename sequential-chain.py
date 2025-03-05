from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

prompt_str = "Please give name, cast and year of release for movies similar to the movie {movie_name}"
prompt_tmpl = PromptTemplate(prompt_str)

llm = Ollama(model="qwen2.5:latest")

p = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)

response = p.run(movie_name="The Matrix")

print(response)