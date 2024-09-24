"""# Text **AI**"""



import os

# Set your Hugging Face API token directly
HUGGINGFACEHUB_API_TOKEN = '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'

# Set the environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN



!pip install langchain-community # install the module
!pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, LLMChain

import pathlib
import textwrap
from IPython.display import display, Markdown

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Setup HuggingFace Access Token
import os
from getpass import getpass

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Import document
from google.colab import drive
drive.mount('/content/drive')

loader = PyPDFDirectoryLoader("/content/drive/MyDrive/Pdf")
docs = loader.load()

# Text Splitting - Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Vector Store - FAISS or ChromaDB
vectorstore = Chroma.from_documents(chunks, embeddings)

query = "Result"
search = vectorstore.similarity_search(query)
to_markdown(search[0].page_content)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
retriever.get_relevant_documents(query)

# Large Language Model - Open Source
llm = LlamaCpp(
    model_path= "/content/drive/MyDrive/Pdf/Bio/BioMistral-7B.Q4_K_M.gguf",
    temperature=0.3,
    max_tokens=2048,
    top_p=1
)

# RAG Chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

template = """
<|context|>
You are an AI assistant that follows instruction extremely well.
Please be truthful and give direct answers
</s>
<|user|>
{query}
</s>
 <|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Interactive Loop
# while True:
#     user_input = input(f"Input Prompt: ")
#     if user_input == 'exit':
#         print('Exiting')

#         sys.exit()
#     if user_input == '':
#         continue
#     result = rag_chain.invoke(user_input)
#     print("Answer: ", result)

import sys
from translate import Translator

# Initialize the translator for English to Telugu
translator = Translator(to_lang="te")

while True:
    user_input = input(f"Input Prompt: ")

    if user_input.lower() == 'exit':
        print('Exiting')
        sys.exit()

    if user_input == '':
        continue

    # Translate user input to Telugu
    translated_input = translator.translate(user_input)
    #print(f"Translated to Telugu: {translated_input}")

    # Process the input with rag_chain (replace with your actual rag_chain implementation)
    result = rag_chain.invoke(user_input)

    # Translate the result to Telugu (assuming rag_chain returns the result in English)
    translated_result = translator.translate(result)

    # Print the final answer in Telugu
    print("Answer in Telugu: ", translated_result)

# Interactive Loop
while True:
    user_input = input(f"Input Prompt: ")
    if user_input == 'exit':
        print('Exiting')

        sys.exit()
    if user_input == '':
        continue
    result = rag_chain.invoke(user_input)
    print("Answer: ", result)
