from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
from langchain.chains.sql_database.query import create_sql_query_chain

model = OllamaLLM(model="mistral:latest", temperature=0)

host = "localhost"
port = "3306"
username="root"
password="YOUR_DB_PASSWORD"

database_scheme = "YOUR_DB_NAME"
mysql_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_scheme}"
db = SQLDatabase.from_uri(mysql_uri, sample_rows_in_table_info=2)
chain = create_sql_query_chain(model, db)

print(db.dialect)
print(db.get_usable_table_names())

# print(db.run("SELECT count(*) FROM argonz_posts LIMIT 10;"))

response = chain.invoke({"question": "How many posts are there"})
print(response)

db.run(response)