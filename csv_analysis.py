import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_ollama import OllamaLLM
import streamlit as st

def main():
    df = pd.read_csv('HR-Employee-Attrition.csv')
    st.set_page_config(
        page_title="Documentation Chatbot",
        page_icon="📚"
    )
    
    st.title("Attrition Analysis Chatbot")
    st.subheader("Uncover Insights from Attrition Data!")
    st.markdown(
    """
    This chatbot was created to answer questions from a set of Attrition data from your organization.
    Ask a question and the chatbot will respond with appropriate analysis.
    """
    )
    
    st.write(df.head())
    user_question = st.text_input("Ask your question about the data..")
    
    agent = create_pandas_dataframe_agent(
        OllamaLLM(model="mistral:latest", temperature=0),
        df=df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,  # Allow execution of potentially dangerous code
        kwargs={
            " handle_parsing_errors": True
        }  # Handle parsing errors gracefully
    )
    
    if user_question:
        try:
            # Use a custom prompt to guide the agent's output
            answer = agent.invoke(f"""
                You are a data analyst. Your task is to analyze the dataset and provide a clear, concise answer to the following question:
                {user_question}
                
                Rules:
                1. Always provide a final answer in a structured format.
                2. Do not include intermediate steps or actions in the final answer.
                3. If you need to perform calculations, do so silently and only return the final result.
            """)
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()