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

# Inicializar modelo de embeddings
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

st.markdown(
    """
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #ECEFF1;
        color: #263238;
    }
    .chat-message.assistant {
        background-color: #E8F5E9;
        color: #1B5E20;
    }
    .chat-message .avatar {
        width: 32px;
        height: 32px;
        margin-right: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        line-height: 32px;
        font-size: 1.5rem;
    }
    .chat-message .content {
        max-width: calc(100% - 50px);
    }
    .chat-message .header {
        font-size: 0.85rem; 
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #424242;
    }
    .sources-section {
        margin-top: 1rem;
        border-top: 1px solid #EEEEEE;
        padding-top: 0.5rem;
    }
    .source-item {
        background-color: #F5F5F5;
        border-radius: 0.3rem;
        padding: 0.5rem;
        margin-bottom: 0.3rem;
        font-size: 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

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
        st.rerun()

st.header("🐕 Pakkun - Assistente de Código")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    # Mensagem de boas-vindas
    welcome_message = """Oi! Eu sou o Pakkun, seu assistente de código. Como posso ajudar você hoje? 
Faça perguntas sobre o código indexado e eu farei o meu melhor para responder!"""
    st.session_state["chat_history"].append(AIMessage(content=welcome_message))

chat_container = st.container()

with st.container():
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "Digite sua pergunta sobre o código:",
            placeholder="Por exemplo: Como funciona a função X?",
            label_visibility="collapsed",
            key="user_question",
        )
    with col2:
        send_button = st.button("Enviar", use_container_width=True)

if "input_to_process" not in st.session_state:
    st.session_state["input_to_process"] = None

if user_input and send_button:
    st.session_state["input_to_process"] = user_input
    st.rerun()

if st.session_state.get("input_to_process"):
    user_input = st.session_state["input_to_process"]

    st.session_state["chat_history"].append(HumanMessage(content=user_input))

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

    st.session_state["input_to_process"] = None
    st.rerun()

with chat_container:
    for idx, message in enumerate(st.session_state["chat_history"]):
        if isinstance(message, HumanMessage):
            st.markdown(
                f"""
            <div class="chat-message user">
                <div class="avatar">👤</div>
                <div class="content">
                    <div class="header">Você</div>
                    {message.content}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif isinstance(message, AIMessage):
            sources_html = ""

            if (
                "current_sources" in st.session_state
                and idx in st.session_state["current_sources"]
            ):
                sources = st.session_state["current_sources"][idx]
                if sources:
                    sources_html = (
                        '<div class="sources-section"><strong>Fontes:</strong>'
                    )
                    for doc in sources:
                        source = doc.metadata.get("source", "Desconhecido")
                        sources_html += f'<div class="source-item">📄 {source}</div>'
                    sources_html += "</div>"

            st.markdown(
                f"""
            <div class="chat-message assistant">
                <div class="avatar">🐕</div>
                <div class="content">
                    <div class="header">Pakkun</div>
                    {message.content}
                    {sources_html}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
