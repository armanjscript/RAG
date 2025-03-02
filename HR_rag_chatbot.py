from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

loader = WebBaseLoader("https://www.wikihow.com/Do-Yoga")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)

llm = OllamaLLM(model="mistral:latest", temperature=0)

embedding = OllamaEmbeddings(model="nomic-embed-text:latest", num_gpu=1)

vectorstore = FAISS.from_documents(documents=documents, embedding=embedding)

retriever = vectorstore.as_retriever()

prompt_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"),
])

retriever_chain = create_history_aware_retriever(llm, retriever=retriever, prompt=prompt_template)

sample_answer = """Some key points for Yoga beginners are: 
        1. Find a comfortable place and time to practice. 
        2. Set a routine that suits you. 
        and so on.. 
        ."""
chat_history = [HumanMessage(content="What are the key things to consider for someone starting to practice Yoga?"),
                AIMessage(content=sample_answer)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

output = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Can you elaborate the first point?",
})

print(output["answer"])

