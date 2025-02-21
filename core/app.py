import os
import re

import streamlit as st
import torch
from dotenv import load_dotenv
from langchain.callbacks import LangChainTracer
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

load_dotenv()

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
INDEX_NAME = "codebase-index"


def set_theme():
    st.markdown(
        """
        <style>
        /* ... (mantido igual) ... */
        </style>
        """,
        unsafe_allow_html=True,
    )


def carrega_modelo(folder_path):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    metadata_filter = {}

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5, "filter": metadata_filter}
    )

    system_message = """Você é Pakkun, especialista em código. Use APENAS estes contextos:

{context}

Regras de resposta:
1. Priorize arquivos mencionados explicitamente
2. Referencie metadados (ex: `source: path/arquivo.py`)
3. Seja técnico e detalhista"""

    template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = (
        RunnableLambda(
            lambda x: {
                **x,
                "context": "\n\n".join(
                    [
                        f"{doc.metadata.get('source', 'Unknown')}:\n{doc.page_content[:300]}"
                        for doc in retriever.get_relevant_documents(x["input"])
                    ]
                ),
            }
        )
        | template
        | ChatGroq(model_name="deepseek-r1-distill-llama-70b")
    )

    st.session_state["chain"] = chain


tracer = LangChainTracer(
    project_name="PakkunMonitor", tags=["production", "code-analysis"]
)


def split_thought_and_response(text):

    if not isinstance(text, str):
        text = getattr(text, "content", str(text))
    pattern = r"<think>(.*?)</think>(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        internal_thought = match.group(1).strip()
        final_response = match.group(2).strip()
        return internal_thought, final_response
    else:
        return "", text.strip()


def chat():
    st.header("Oi eu sou Pakkun", divider=True)

    if "chain" not in st.session_state:
        st.error("Carregue o Pakkun primeiro")
        st.stop()

    memoria = st.session_state.get("memoria", ConversationBufferMemory())

    for msg in memoria.buffer_as_messages:
        st.chat_message(msg.type).markdown(msg.content)

    if input_user := st.chat_input("Fala comigo!"):
        arquivos_relevantes = extrair_metadados(input_user)
        adjusted_filter = (
            {"source": {"$in": arquivos_relevantes}} if arquivos_relevantes else {}
        )

        st.chat_message("human").markdown(input_user)

        with st.spinner("Carregando resposta..."):
            resposta = st.session_state["chain"].invoke(
                {"input": input_user, "chat_history": memoria.buffer_as_messages},
                config={"callbacks": [tracer]},
            )

        internal_thought, final_response = split_thought_and_response(resposta)

        combined_message = ""
        if internal_thought:
            combined_message += (
                f"<details>\n"
                f"<summary>Pensamento interno (minimizado)</summary>\n\n"
                f"{internal_thought}\n"
                f"</details>\n\n"
            )
        combined_message += final_response

        st.chat_message("ai").markdown(combined_message, unsafe_allow_html=True)

        memoria.chat_memory.add_user_message(input_user)
        memoria.chat_memory.add_ai_message(final_response)
        st.session_state["memoria"] = memoria


def extrair_metadados(input_text):
    padrao_arquivos = r"\b[\w\-_]+\.(py|js|ts|md|txt)\b"
    matches = re.findall(padrao_arquivos, input_text)
    base_dir = st.session_state.get("folder_path", "")
    arquivos_relevantes = []

    for nome_arquivo in matches:
        caminho = os.path.join(base_dir, nome_arquivo)
        if os.path.exists(caminho):
            arquivos_relevantes.append(caminho)

    return arquivos_relevantes


def list_all_subdirectories(base_dir):
    abs_base = os.path.abspath(base_dir)
    subfolders = []
    for root, dirs, files in os.walk(abs_base):
        for d in dirs:
            folder_path = os.path.join(root, d)
            subfolders.append(folder_path)
    parent_folder = os.path.dirname(abs_base)
    if parent_folder and parent_folder != abs_base:
        subfolders.insert(0, parent_folder)
    return subfolders


def main():
    set_theme()

    default_base_dir = ".."
    try:
        folder_list = list_all_subdirectories(default_base_dir)
        selected_folder = folder_list[0] if folder_list else default_base_dir
    except Exception as e:
        st.error(f"Erro ao listar pastas: {e}")
        st.stop()

    st.session_state["folder_path"] = selected_folder
    carrega_modelo(selected_folder)
    st.success("Modelo conectado com a base e pronto!")

    chat()


if __name__ == "__main__":
    main()
