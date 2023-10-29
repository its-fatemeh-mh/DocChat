from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader, TextLoader
import os 
import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma 
import fitz

#streamlit layout
st.set_page_config(page_title="DocChat", layout="wide",)
st.markdown(f"""
            <style>
            .stApp {{background-image: url("https://images.unsplash.com/photo-1509537257950-20f875b03669?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1469&q=80"); 
                     background-attachment: fixed;
                     background-attachment: fixed;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)
         


# Saving a Document
def save_pdf(doc,file_path):
    with open(os.path.join(file_path, doc.name), "wb") as f:
        f.write(doc.read())
    return  st.success(f"PDF file '{doc.name}' has been saved." )   

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def write_text_file(content, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False

# set prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])



# initialize the LLM & Embeddings
llm = LlamaCpp(model_path="./models/llama-7b.ggmlv3.q4_0.bin")
embeddings = LlamaCppEmbeddings(model_path="models/llama-7b.ggmlv3.q4_0.bin")
llm_chain = LLMChain(llm=llm, prompt=prompt)


st.title("Q & A with Doucuments")
doc = st.file_uploader("Upload a pdf file", type="pdf", accept_multiple_files=False)

if doc is not None:
    file_path = "temp/file.txt"
    text_data = get_pdf_text(doc)
    write_text_file(text_data, file_path) 
    loader = TextLoader(file_path)
    text = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(text)
    vectordb = Chroma.from_documents(texts, embeddings)
    st.success("File has been loaded successfully!")

#Quering through LLM
question = st.text_input("Ask something from the file", disabled=not doc)
if question:
    similar_page = vectordb.similarity_search(question, k=1)
    context = similar_page[0].page_content
    query_llm = LLMChain(llm=llm, prompt=prompt)
    response = query_llm.run({"context": context, "question": question})        
    st.write(response)




