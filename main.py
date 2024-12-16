import os
from io import StringIO
import pandas as pd
import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tempfile




# LLM and key loading function
def load_LLM(openai_api_key):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm

# Page title and header
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")

# Intro: instructions
col1, col2 = st.columns(2)
with col1:
    st.markdown("Summarize long texts, PDFs, or CSV files with AI.")
with col2:
    st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")

# Input OpenAI API Key
st.markdown("## Enter Your OpenAI API Key")
def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text

openai_api_key = get_openai_api_key()

# File upload
st.markdown("## Upload your file")
uploaded_file = st.file_uploader("Upload your file (TXT, PDF, or CSV)", type=["txt", "pdf", "csv"])

# Output summary
st.markdown("### Here is your Summary:")

if uploaded_file is not None:
    file_input = ""
    # Determine file type by extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    # Process file based on type
    if file_extension == ".txt":
        file_input = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    elif file_extension == ".pdf":
        # Save PDF file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Load the PDF using PyPDFLoader
        pdf_loader = PyPDFLoader(temp_file_path)
        file_input = " ".join([page.page_content for page in pdf_loader.load()])
    elif file_extension == ".csv":
        dataframe = pd.read_csv(uploaded_file)
        file_input = dataframe.to_string()
    else:
        st.error("Unsupported file type. Please upload a TXT, PDF, or CSV file.")
        st.stop()

    # Validate file input length
    if len(file_input.split(" ")) > 20000:
        st.write("Please enter a shorter file. The maximum length is 20,000 words.")
        st.stop()

    if file_input:
        if not openai_api_key:
            st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
            st.stop()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=350)
        splitted_documents = text_splitter.create_documents([file_input])

        # Load LLM and summarize
        llm = load_LLM(openai_api_key=openai_api_key)
        summarize_chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
        summary_output = summarize_chain.run(splitted_documents)

        # Display the summary
        st.write(summary_output)
