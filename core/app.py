import streamlit as st
import os
import torch
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool


load_dotenv()

# Configurações
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

# Define os AttributeInfo baseados nos metadados dos seus documentos
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]


document_content_description = "All codes from Django/React app"

retriever = SelfQueryRetriever.from_llm(
    llm,
    chroma,
    document_content_description,
    metadata_field_info,
    verbose=True,
)

# Cria um template de prompt para a resposta
system_prompt = (
    "Use o contexto fornecido para responder à pergunta. "
    "Se não souber, diga que não sabe. "
    "Responda de forma completa e informativa. "
    "Contexto: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Cria a chain que combina os documentos recuperados
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Cria a retrieval chain usando o retriever (self-querying) e a chain de QA
qa_chain = create_retrieval_chain(retriever, question_answer_chain)

# Configuração do Streamlit
st.set_page_config(
    page_title="Pakkun - Assistente de Código",
    page_icon="🐕",
    layout="centered",
    initial_sidebar_state="expanded",
)


def display_message(msg):

    if isinstance(msg, dict):
        answer = msg.get("answer", "")

        if "<think>" in answer and "</think>" in answer:
            start = answer.find("<think>")
            end = answer.find("</think>")
            prefix = answer[:start].strip()
            think_content = answer[start + len("<think>") : end].strip()
            suffix = answer[end + len("</think>") :].strip()
            if prefix:
                st.markdown(prefix)
            with st.expander("Detalhes", icon="🧠"):
                st.info(think_content)
            if suffix:
                st.markdown(suffix)
        else:
            st.markdown(answer)

        # Se existir contexto, exibe as fontes (por exemplo, o campo "source" de cada Document)
        context_docs = msg.get("context", [])
        if context_docs:
            st.markdown("📄 Fontes")
            for doc in context_docs:
                source = doc.metadata.get("source", "Desconhecido")
                st.success("- " + source)
    elif isinstance(msg, str):
        if "<think>" in msg and "</think>" in msg:
            start = msg.find("<think>")
            end = msg.find("</think>")
            prefix = msg[:start].strip()
            think_content = msg[start + len("<think>") : end].strip()
            suffix = msg[end + len("</think>") :].strip()
            if prefix:
                st.markdown(prefix)
            with st.expander("Detalhes", icon="🧠"):
                st.info(think_content)
            if suffix:
                st.markdown(suffix)
        else:
            st.markdown(msg)


# Sidebar com instruções e exemplos
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

# Inicializa o histórico de conversa
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    welcome_message = (
        "Oi! Eu sou o Pakkun, seu assistente de código. Como posso ajudar você hoje? "
        "Faça perguntas sobre o código indexado e vamos juntos melhorar e corrigir seu código!"
    )
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": welcome_message}
    )

chat_container = st.container()
user_input = st.chat_input("Digite sua pergunta sobre o código:")


def render_chat():
    for message in st.session_state["chat_history"]:
        if message["role"] == "user":
            with st.chat_message("user"):
                display_message(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                display_message(message["content"])


if user_input:
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    render_chat()

    with st.spinner("Pakkun está pensando..."):
        result = qa_chain.invoke({"input": user_input})
        # O resultado pode vir com a chave "output" ou direto
        response = result.get("output", result)
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": response}
        )

    render_chat()
