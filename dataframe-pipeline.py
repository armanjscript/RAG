from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent
)
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
import pandas as pd
from llama_index.experimental.query_engine.pandas import PandasInstructionParser

df = pd.read_csv('ObesityDataset.csv', encoding='latin-1')
# print(df)

instruction_str = (
    "1. Convert the query to execute Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PPRINT ONLY THE EXPRESSION.\n"
    "5. Do not use quotes in the expression.\n"
)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)

response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response:"
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, 
    df_str=df.head(5)
)

pandas_output_str = PandasInstructionParser(df)

response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = Ollama(model="qwen2.5:latest")

#Build Query Pipeline

qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_str,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm
    },
    verbose=True
)

#The Pipeline
qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links([
    Link("input", "response_synthesis_prompt", dest_key="query_str"),
    Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
    Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output")
])

qp.add_link("response_synthesis_prompt", "llm2")


response = qp.run(
    # query_str="What is the distribution between males and females?"
    # query_str="What is the correlation between vegetable Intake and Obesity level? Show the correlation coefficient along with explanation."
    query_str="What is the distribution of people consuming alchol and not?"
)

print(response)