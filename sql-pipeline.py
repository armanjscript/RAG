from sqlalchemy import (
    create_engine,
    text
)
from llama_index.core import SQLDatabase, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema
)

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")

db_host = "localhost:3306"
db_user="YOUR_USERNAME"
db_password="YOUR_DB_PASSWORD"
db_name = "YOUR_DB_NAME"

mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

engine = create_engine(mysql_uri)

print("Printing three rows:")
with engine.connect() as connection:
    result = connection.execute(text("SELECT * FROM your_table LIMIT 3"))
    for row in result:
        print(row)
        
print("Printing Table structures:")
with engine.connect() as connection:
    result = connection.execute(text("DESCRIBE your_table"))
    for row in result:
        print(row)
        
llm = Ollama(model="mistral:latest", temperature=0.1)

Settings.llm=llm

sql_database = SQLDatabase(engine, include_tables=["your_table"])

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["your_table"],
)

query_str = "How many posts are here?"
# query_str = "How many unique stores are there?"

response = query_engine.query(query_str)

print(response)


table_node_mapping = SQLTableNodeMapping(sql_database=sql_database)
table_schema_objs = [
    SQLTableSchema(table_name="your_table")
]

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    index_cls=VectorStoreIndex
)

query_engine = SQLTableRetrieverQueryEngine(
    sql_database=sql_database,
    table_retriever=obj_index.as_retriever(similarity_top_k=3)
)

query_str = "How many posts are here? Order them by Date."

response = query_engine.query(query_str)
print(response)