import streamlit as st
import os
import torch
import time
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "code_collection"
LLM_MODEL = "deepseek-r1-distill-llama-70b"

device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

chroma = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)

llm = ChatGroq(model_name=LLM_MODEL)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)

retriever = chroma.as_retriever(
    search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer",
)

st.set_page_config(
    page_title="Pakkun - Assistente de Código",
    page_icon="🐕",
    layout="centered",
    initial_sidebar_state="expanded",
)


def display_message(text: str):
    if "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>")
        prefix = text[:start].strip()
        think_content = text[start + len("<think>") : end].strip()
        suffix = text[end + len("</think>") :].strip()

        if prefix:
            st.markdown(prefix)
        with st.expander("Detalhes", icon="🧠"):
            st.info(think_content)
        if suffix:
            st.markdown(suffix)
    else:
        st.markdown(text)


with st.sidebar:
    st.title("🐕 Pakkun")
    st.subheader("Assistente de Código Inteligente")
    st.markdown("### Como usar:")
    st.markdown(
        """
        1. **Faça perguntas específicas** sobre o código indexado  
        2. **Peça explicações** sobre partes complexas do código  
        3. **Solicite recomendações** para melhorar seu código  
        4. **Pergunte sobre padrões** ou boas práticas  
        """
    )
    st.markdown("### Exemplos de perguntas:")
    st.info("Como funciona a função X no arquivo Y?")
    st.info("Qual é o propósito da classe Z?")
    st.info("Como posso melhorar a performance deste trecho de código?")

    if st.button("Limpar conversa"):
        st.session_state["chat_history"] = []
        st.session_state["current_sources"] = {}
        st.rerun()

st.header("🐕 Pakkun - Assistente de Código")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    welcome_message = (
        "Oi! Eu sou o Pakkun, seu assistente de código. Como posso ajudar você hoje? "
        "Faça perguntas sobre o código indexado e eu farei o meu melhor para responder!"
    )
    st.session_state["chat_history"].append(AIMessage(content=welcome_message))

chat_container = st.container()

user_input = st.chat_input("Digite sua pergunta sobre o código:")


def render_chat():
    chat_placeholder = st.container()
    for idx, message in enumerate(st.session_state["chat_history"]):
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                display_message(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                display_message(message.content)
                if (
                    "current_sources" in st.session_state
                    and idx in st.session_state["current_sources"]
                ):
                    st.markdown("📄 Fontes")
                    for doc in st.session_state["current_sources"][idx]:
                        st.success("- " + doc.metadata.get("source", "Desconhecido"))


if user_input:

    st.session_state["chat_history"].append(HumanMessage(content=user_input))
    render_chat()

    with st.spinner("Pakkun está pensando..."):
        result = qa_chain.invoke({"question": user_input})
        response = result["answer"]
        sources = result.get("source_documents", [])
        st.session_state["chat_history"].append(AIMessage(content=response))
        if "current_sources" not in st.session_state:
            st.session_state["current_sources"] = {}
        st.session_state["current_sources"][
            len(st.session_state["chat_history"]) - 1
        ] = sources

    render_chat()
