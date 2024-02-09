from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from decouple import config
import streamlit as st


from langchain.embeddings import OpenAIEmbeddings
embedding=OpenAIEmbeddings(openai_api_key= st.secrets.OPENAI_API_KEY)
vectordb = Chroma(
    persist_directory= "C:/Users/USER/Downloads/Retrival_methods/new_chroma_sop/db",
    embedding_function=embedding,
    
)

chroma_retriever = Chroma()
from langchain.chat_models import ChatOpenAI

llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",openai_api_key= st.secrets.OPENAI_API_KEY)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm= llm,
    retriever= vectordb.as_retriever(),
    return_source_documents=True,
    chain_type="stuff"
   )
    


def rag_func(question: str) ->str:
    """
    This function takes in user question or prompt and returns a response
    :param: question: string valiue of the question or the prompt from the user
    :returns: string value of the answer to the user question
    """
    response = qa_chain({"query": question})

    return response


# def process_llm_response(llm_response):
#     return llm_response['result']
#     for source in llm_response["source_documents"]:
#         return source.metadata['source']
#     # return llm_response['result']
#     print('\n\nSources:')
#     # for source in llm_response["source_documents"]:
#     #     return source.metadata['source']


def process_llm_response(llm_response):
    result = llm_response['result']
    for source in llm_response["source_documents"]:
        return result, source.metadata['source']

