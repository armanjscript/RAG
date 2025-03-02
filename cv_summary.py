from langchain_community.document_loaders import Docx2txtLoader
from langchain_ollama import OllamaLLM
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

import tempfile
import streamlit as st

def process_docx(docx_file):
    loader = Docx2txtLoader(docx_file)
    text = loader.load()
    return text

def process_pdf(pdf_file):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load the PDF from the temporary file
    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()
    
    text = ""
    for page in pages:
        text += page.page_content
    
    text = text.replace('\t', ' ')
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=50
    )
    
    texts = text_splitter.create_documents([text])
    
    return texts

def main():
    st.title("CV Summary Generator")
    
    uploaded_file = st.file_uploader("Select CV", type=["docx", "pdf"])
    
    text = ""
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        
        st.write("File Details:")
        st.write(f"File Name: {uploaded_file.name}")
        st.write(f"File Type: {file_extension}")
        
        if file_extension == "docx":
            text = process_docx(uploaded_file)
        elif file_extension == "pdf":
            text = process_pdf(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a .docx or .pdf file.")
            return
        
        llm = OllamaLLM(model="mistral:latest", temperature=0)
        prompt_template = """You have been given a Resume to analyze. 
        Write a verbose detail of the following: 
        {text} 
        Details:"""
        prompt = PromptTemplate.from_template(template=prompt_template)
        
        refine_template = (
            "Your job is to produce a final outcome\n"
            "We have provided an existing detail: {existing_answer}\n"
            "We want a refined version of the existing detail based on initial details below\n"
            "--------------\n"
            "{text}\n"
            "--------------\n"
            "Given the new context, refine the original summary in the following manner:"
            "Name: \n"
            "Email: \n"
            "Key Skills: \n"
            "Last Company: \n"
            "Experience summary: \n"
        )
        
        refine_prompt = PromptTemplate.from_template(template=refine_template)
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text"
        )
        
        result = chain.invoke({"input_documents": text}, return_only_outputs=True)
        st.write("Resume Summary:")
        st.text_area("Text", result["output_text"], height=300)

if __name__ == "__main__":
    main()