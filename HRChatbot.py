import os
import requests
from openai import OpenAI
import streamlit as st

# import torch
# import torchaudio
import torch
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

os.environ['OPENAI_API_KEY']= "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"

@st.cache_resource
def create_query_engine():
    # documents= SimpleDirectoryReader("/content/drive/MyDrive/Colab Notebooks/LLMs/HR Chatbot").load_data()
    print('Current Working Directory - ', os.getcwd())
    print('List Directory - ', os.listdir())

    documents= SimpleDirectoryReader("./Documents").load_data()
    llm = ChatOpenAI(temperature=0, max_tokens=800,
                     model_kwargs={"top_p": 0, "frequency_penalty": 0, "presence_penalty": 0})
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    service_context = ServiceContext.from_defaults(chunk_size=512, llm=llm, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    # Initialize the reranker
    rerank = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=7)
    query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[rerank]) # Note we are first selecting 10 chunks.
    # query_engine = index.as_query_engine(similarity_top_k=10) # Note we are first selecting 10 chunks.

    return query_engine

# def get_response(prompt):    
#     response= query_engine()
#     return response

query_eng= create_query_engine()


st.title("💬 MOHRE Conversational Chatbot - UAE Labour Laws")
st.caption("🚀 A streamlit chatbot powered by OpenAI with RAGs")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner(text="In progress...", cache=False):
      response = query_eng.query(prompt)
    
    msg = response.response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)




