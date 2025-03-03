# import os
# import re
#
# import streamlit as st
# import torch
# from dotenv import load_dotenv
# from langchain.callbacks import LangChainTracer
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.runnables import RunnableLambda
# from langchain_groq import ChatGroq
# from langchain_pinecone import PineconeVectorStore
#
# load_dotenv()
#
# EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
# INDEX_NAME = "codebase-index"
#
#
# def set_theme():
#     st.markdown(
#         """
#         <style>
#         /* ... (mantido igual) ... */
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )
#
#
# def carrega_modelo(folder_path):
#     embeddings = HuggingFaceEmbeddings(
#         model_name=EMBEDDINGS_MODEL,
#         model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
#         encode_kwargs={"normalize_embeddings": True},
#     )
#
#     vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
#
#     metadata_filter = {}
#
#     retriever = vectorstore.as_retriever(
#         search_type="similarity", search_kwargs={"k": 5, "filter": metadata_filter}
#     )
#
#     system_message = """Você é Pakkun, especialista em código. Use APENAS estes contextos:
#
# {context}
#
# Regras de resposta:
# 1. Priorize arquivos mencionados explicitamente.
# 2. Referencie metadados (ex: `source: path/arquivo.py` ou `source: path/arquivo.py (session 2)`).
# 3. Seja técnico e detalhista."""
#
#     template = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_message),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )
#
#     # Atualiza o contexto para incluir, se disponível, o session_index
#     chain = (
#         RunnableLambda(
#             lambda x: {
#                 **x,
#                 "context": "\n\n".join(
#                     [
#                         f"{doc.metadata.get('source', 'Unknown')}"
#                         + (
#                             f" (session {doc.metadata.get('session_index')})"
#                             if "session_index" in doc.metadata
#                             else ""
#                         )
#                         + f":\n{doc.page_content[:300]}"
#                         for doc in retriever.get_relevant_documents(x["input"])
#                     ]
#                 ),
#             }
#         )
#         | template
#         | ChatGroq(model_name="deepseek-r1-distill-llama-70b")
#     )
#
#     st.session_state["chain"] = chain
#
#
# tracer = LangChainTracer(
#     project_name="PakkunMonitor", tags=["production", "code-analysis"]
# )
#
#
# def split_thought_and_response(text):
#     if not isinstance(text, str):
#         text = getattr(text, "content", str(text))
#     pattern = r"<think>(.*?)</think>(.*)"
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         internal_thought = match.group(1).strip()
#         final_response = match.group(2).strip()
#         return internal_thought, final_response
#     else:
#         return "", text.strip()
#
#
# def chat():
#     st.header("Oi eu sou Pakkun", divider=True)
#
#     if "chain" not in st.session_state:
#         st.error("Carregue o Pakkun primeiro")
#         st.stop()
#
#     memoria = st.session_state.get("memoria", ConversationBufferMemory())
#
#     for msg in memoria.buffer_as_messages:
#         st.chat_message(msg.type).markdown(msg.content)
#
#     if input_user := st.chat_input("Fala comigo!"):
#         arquivos_relevantes = extrair_metadados(input_user)
#         adjusted_filter = (
#             {"source": {"$in": arquivos_relevantes}} if arquivos_relevantes else {}
#         )
#
#         st.chat_message("human").markdown(input_user)
#
#         with st.spinner("Carregando resposta..."):
#             resposta = st.session_state["chain"].invoke(
#                 {"input": input_user, "chat_history": memoria.buffer_as_messages},
#                 config={"callbacks": [tracer]},
#             )
#
#         internal_thought, final_response = split_thought_and_response(resposta)
#
#         combined_message = ""
#         if internal_thought:
#             combined_message += (
#                 f"<details style='color: grey; font-size: medium;'>\n"
#                 f"<summary style='font-size: medium;'>Raciocínio</summary>\n\n"
#                 f"{internal_thought}\n"
#                 f"</details>\n\n"
#             )
#         combined_message += final_response
#
#         st.chat_message("ai").markdown(combined_message, unsafe_allow_html=True)
#
#         memoria.chat_memory.add_user_message(input_user)
#         st.session_state["memoria"] = memoria
#
#
# def extrair_metadados(input_text):
#     # Regex aprimorada para capturar caminhos relativos (ex: src/app.py)
#     padrao_arquivos = r"[\w\-.\\/]+\.(?:py|js|ts|md|txt)"
#     matches = re.findall(padrao_arquivos, input_text)
#     base_dir = st.session_state.get("folder_path", "")
#     arquivos_relevantes = []
#
#     for nome_arquivo in matches:
#         caminho = os.path.join(base_dir, nome_arquivo)
#         if os.path.exists(caminho):
#             arquivos_relevantes.append(caminho)
#         elif os.path.exists(nome_arquivo):
#             arquivos_relevantes.append(nome_arquivo)
#
#     return arquivos_relevantes
#
#
# def list_all_subdirectories(base_dir):
#     abs_base = os.path.abspath(base_dir)
#     subfolders = []
#     for root, dirs, files in os.walk(abs_base):
#         for d in dirs:
#             folder_path = os.path.join(root, d)
#             subfolders.append(folder_path)
#     parent_folder = os.path.dirname(abs_base)
#     if parent_folder and parent_folder != abs_base:
#         subfolders.insert(0, parent_folder)
#     return subfolders
#
#
# def main():
#     set_theme()
#     default_base_dir = ".."
#     try:
#         folder_list = list_all_subdirectories(default_base_dir)
#         selected_folder = folder_list[0] if folder_list else default_base_dir
#     except Exception as e:
#         st.error(f"Erro ao listar pastas: {e}")
#         st.stop()
#
#     st.session_state["folder_path"] = selected_folder
#     carrega_modelo(selected_folder)
#     st.success("Modelo conectado com a base e pronto!")
#     chat()
#
#
# if __name__ == "__main__":
#     main()
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Any, Union

import streamlit as st
import torch
from dotenv import load_dotenv
from langchain.callbacks import LangChainTracer
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

# Carrega variáveis de ambiente
load_dotenv()

# Configurações globais
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "code_collection"
LLM_MODEL = "deepseek-r1-distill-llama-70b"
MAX_CONTEXT_LENGTH = 3500  # Número máximo de caracteres por documento no contexto
MAX_CONTEXT_DOCS = 8  # Número máximo de documentos no contexto

# Cores e temas - usando cores padrão do Streamlit para melhor visibilidade
THEME_COLORS = {
    "primary": "#ff4b4b",       # Vermelho Streamlit
    "secondary": "#0068c9",     # Azul Streamlit
    "background": "#ffffff",    # Fundo branco
    "text": "#262730",          # Texto escuro padrão
    "code_bg": "#f0f2f6",       # Fundo de código claro
    "success": "#09ab3b",       # Verde Streamlit
    "warning": "#ffbd45",       # Amarelo/laranja Streamlit
    "error": "#ff4b4b",         # Vermelho Streamlit
}


def configure_page():
    """Configura título e aparência da página."""
    st.set_page_config(
        page_title="Pakkun - Assistente de Código",
        page_icon="🐕",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Ajusta estilo da página com CSS minimalista para garantir legibilidade
    st.markdown(
        """
        <style>
        pre {
            padding: 10px;
            border-radius: 5px;
            background-color: #f0f2f6;
        }
        code {
            font-family: 'Courier New', monospace;
        }
        .source-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 5px 10px;
            margin: 5px 0;
            background-color: #f8f8f8;
            font-size: 0.8em;
        }
        details {
            margin: 10px 0;
            padding: 10px;
            background-color: #f0f2f6;
            border-radius: 5px;
        }
        summary {
            font-weight: bold;
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


class RAGAssistant:
    """
    Implementa um assistente de código baseado em técnicas RAG (Retrieval Augmented Generation).
    
    Funcionalidades:
    - Conexão com ChromaDB para busca vetorial e semântica
    - Filtragem contextual de documentos relevantes
    - Extração de metadados de arquivos mencionados na consulta
    - Reranking de resultados para melhorar relevância
    - Histórico de conversa com memória
    """
    
    def __init__(
        self,
        embeddings_model: str = EMBEDDINGS_MODEL,
        persist_dir: str = PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        llm_model: str = LLM_MODEL,
        device: Optional[str] = None,
        max_context_docs: int = MAX_CONTEXT_DOCS,
    ):
        """Inicializa o assistente RAG."""
        self.embeddings_model = embeddings_model
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.max_context_docs = max_context_docs
        
        # Determina o device para embeddings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inicializa embeddings, vectorstore e LLM
        self.initialize_components()
        
        # Configura tracer para monitoramento
        self.tracer = LangChainTracer(
            project_name="PakkunMonitor", 
            tags=["production", "code-analysis", "rag"]
        )
    
    def initialize_components(self):
        """Inicializa os componentes necessários do sistema RAG."""
        # Inicializa o modelo de embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Verifica se o diretório do ChromaDB existe
        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(
                f"Base ChromaDB não encontrada em {self.persist_dir}. "
                "Execute primeiro o code_indexer.py."
            )
        
        # Inicializa o vectorstore
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        
        # Cria o retriever padrão
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.max_context_docs, "filter": None},
        )
        
        # Inicializa o modelo de linguagem
        self.llm = ChatGroq(model_name=self.llm_model)
        
        # Cria a cadeia RAG
        self.chain = self._build_rag_chain()
    
    def _build_rag_chain(self):
        """
        Constrói a cadeia RAG para busca, geração e resposta.
        Implementa técnicas avançadas de RAG para melhorar a qualidade das respostas.
        """
        # System prompt com instruções para o assistente
        system_message = """Você é Pakkun, assistente de código especializado em análise e explicação de codebases.

Use apenas os contextos fornecidos para responder às perguntas:

{context}

Regras obrigatórias:
1. Priorize informações de arquivos explicitamente mencionados na pergunta.
2. Cite as fontes usando o formato: `Fonte: caminho/do/arquivo.ext`.
3. Quando o contexto for insuficiente, diga claramente "Não tenho informações suficientes" em vez de inventar.
4. Seja técnico, preciso e detalhado nas suas explicações de código.
5. Use formatação markdown para destacar trechos de código e conceitos importantes.
6. Se precisar de raciocínio extenso, coloque-o entre tags <think>...</think> - esse conteúdo ficará recolhido na interface.

Você pode usar <think>...</think> para mostrar seu raciocínio detalhado antes da resposta final.

Formatos de resposta:
- Para explicações de código: use blocos de código com syntax highlighting
- Para listar arquivos/funções: use listas com marcadores
- Para mostrar relações: descreva a hierarquia e dependências
"""
        
        # Template da conversa
        template = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                ("system", "Lembre-se de citar as fontes e ser técnico e preciso."),
            ]
        )
        
        # Função para processar contexto e buscar documentos relevantes
        def retrieve_and_process_context(query_bundle):
            """Recupera e processa documentos relevantes com base na consulta."""
            query = query_bundle["input"]
            
            # Extrai nomes de arquivos mencionados na consulta
            mentioned_files = self.extract_file_mentions(query)
            
            # Realiza buscas com diferentes estratégias
            results = self.multi_strategy_retrieval(
                query, 
                mentioned_files=mentioned_files
            )
            
            # Formata o contexto para o modelo
            formatted_context = self.format_context_for_llm(results)
            
            return {
                **query_bundle,
                "context": formatted_context,
                "retrieved_docs": results,
            }
        
        # Monta a cadeia RAG
        chain = (
            RunnableLambda(retrieve_and_process_context)
            | template
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def extract_file_mentions(self, query: str) -> List[str]:
        """
        Extrai menções a arquivos na consulta do usuário.
        Suporta vários formatos de caminhos de arquivo.
        """
        # Regex aprimorada para capturar diversos formatos de caminhos de arquivo
        file_patterns = [
            r'[\w\-.\\/]+\.(?:py|js|jsx|ts|tsx|md|html|css|java|c|cpp|h|go|rs|php|rb|json|yml|yaml|txt|sh)',  # Caminhos de arquivo
            r'["\'](?:[\w\-.\\/]+\.(?:py|js|jsx|ts|tsx|md|html|css|java|c|cpp|h|go|rs|php|rb|json|yml|yaml|txt|sh))["\']',  # Caminhos entre aspas
        ]
        
        mentioned_files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                # Remove aspas se presentes
                clean_match = match.strip('\'"')
                mentioned_files.append(clean_match)
        
        return list(set(mentioned_files))  # Remove duplicados
    
    def multi_strategy_retrieval(
        self, 
        query: str, 
        mentioned_files: Optional[List[str]] = None,
        k_semantic: int = 5,
        k_keyword: int = 3,
        k_structural: int = 2,
    ) -> List[Document]:
        """
        Implementa múltiplas estratégias de recuperação para melhorar a qualidade dos resultados.
        
        1. Busca por arquivos mencionados explicitamente
        2. Busca semântica por similaridade vetorial
        3. Busca por estruturas de código específicas
        4. Combinação e reranking dos resultados
        """
        results = []
        
        # 1. Busca por arquivos mencionados explicitamente
        if mentioned_files:
            metadata_filter = {"source": {"$in": mentioned_files}}
            mentioned_docs = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k_keyword, "filter": metadata_filter},
            ).get_relevant_documents(query)
            results.extend(mentioned_docs)
        
        # 2. Busca semântica padrão (sem filtros)
        semantic_docs = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_semantic, "filter": None},
        ).get_relevant_documents(query)
        results.extend(semantic_docs)
        
        # 3. Busca específica por estruturas de código
        code_structure_filter = {"is_code_structure": {"$eq": True}}
        code_docs = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_structural, "filter": code_structure_filter},
        ).get_relevant_documents(query)
        results.extend(code_docs)
        
        # Busca por chunks específicos se houver menção à "implementação" ou "conteúdo"
        if any(term in query.lower() for term in ["implement", "conteúdo", "código", "função", "método"]):
            chunk_filter = {"is_chunk": {"$eq": True}}
            chunk_docs = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3, "filter": chunk_filter},
            ).get_relevant_documents(query)
            results.extend(chunk_docs)
        
        # Deduplica e reranqueia os resultados
        return self.deduplicate_and_rerank(results, query)
    
    def deduplicate_and_rerank(self, docs: List[Document], query: str) -> List[Document]:
        """
        Deduplica e reranqueia documentos com base em relevância e diversidade.
        """
        # Remove duplicados baseados no conteúdo
        seen_contents = set()
        unique_docs = []
        
        for doc in docs:
            # Usa um hash do conteúdo para verificar duplicação
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)
        
        # Prioriza documentos mencionados explicitamente
        # e mantém diversidade balanceando chunks e documentos completos
        
        # 1. Documentos mencionados no query têm prioridade máxima
        mentioned_docs = []
        other_docs = []
        
        # Extrai nomes de arquivo da consulta
        mentioned_files = self.extract_file_mentions(query)
        
        for doc in unique_docs:
            source = doc.metadata.get("source", "")
            if any(file in source for file in mentioned_files):
                mentioned_docs.append(doc)
            else:
                other_docs.append(doc)
        
        # 2. Separa documentos completos e chunks para garantir diversidade
        complete_docs = [doc for doc in other_docs if not doc.metadata.get("is_chunk", False)]
        chunk_docs = [doc for doc in other_docs if doc.metadata.get("is_chunk", False)]
        
        # 3. Balanceia a combinação para ter uma mistura de diferentes tipos de documentos
        final_docs = []
        final_docs.extend(mentioned_docs)  # Primeiro os documentos mencionados explicitamente
        
        # Alterna entre documentos completos e chunks para diversidade
        remaining_slots = self.max_context_docs - len(mentioned_docs)
        
        if remaining_slots > 0:
            complete_ratio = 0.6  # 60% para documentos completos
            complete_count = min(int(remaining_slots * complete_ratio), len(complete_docs))
            chunk_count = min(remaining_slots - complete_count, len(chunk_docs))
            
            final_docs.extend(complete_docs[:complete_count])
            final_docs.extend(chunk_docs[:chunk_count])
            
            # Se ainda houver slots, adiciona mais documentos
            remaining = self.max_context_docs - len(final_docs)
            if remaining > 0 and complete_count < len(complete_docs):
                final_docs.extend(complete_docs[complete_count:complete_count+remaining])
            
            remaining = self.max_context_docs - len(final_docs)
            if remaining > 0 and chunk_count < len(chunk_docs):
                final_docs.extend(chunk_docs[chunk_count:chunk_count+remaining])
        
        return final_docs[:self.max_context_docs]
    
    def format_context_for_llm(self, docs: List[Document]) -> str:
        """
        Formata documentos recuperados em um formato otimizado para o contexto do LLM.
        Inclui metadados importantes e limita o tamanho para evitar exceder o contexto.
        """
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            # Extrai metadados importantes
            source = doc.metadata.get("source", "Unknown")
            doc_type = doc.metadata.get("file_type", "Unknown")
            is_chunk = doc.metadata.get("is_chunk", False)
            entity_type = doc.metadata.get("entity_type", "")
            entity_name = doc.metadata.get("entity_name", "")
            
            # Prepara cabeçalho com informações do documento
            header = f"[Doc {i}] {source}"
            if entity_type and entity_name:
                header += f" - {entity_type}: {entity_name}"
            elif is_chunk:
                header += " (chunk)"
            
            # Limita o tamanho do conteúdo para evitar contextos muito grandes
            content = doc.page_content
            if len(content) > MAX_CONTEXT_LENGTH:
                content = content[:MAX_CONTEXT_LENGTH] + "...[truncado]"
            
            # Formata o documento
            formatted_doc = f"{header}\n{content}\n"
            context_parts.append(formatted_doc)
        
        # Junta os documentos formatados
        return "\n---\n".join(context_parts)
    
    def process_query(
        self, 
        query: str, 
        chat_history: Optional[List[Union[HumanMessage, AIMessage]]] = None
    ) -> str:
        """
        Processa a consulta do usuário e gera uma resposta.
        Usa o histórico de conversa para contexto adicional.
        """
        if chat_history is None:
            chat_history = []
        
        # Executa a cadeia RAG
        response = self.chain.invoke(
            {"input": query, "chat_history": chat_history},
            config={"callbacks": [self.tracer]}
        )
        
        return response


def extract_thought_and_response(text: str) -> Tuple[str, str]:
    """
    Extrai pensamento interno e resposta final do texto.
    O formato é: <think>pensamento</think>resposta
    """
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


def get_folder_path() -> str:
    """Determina o caminho da pasta principal para indexação."""
    default_base_dir = ".."
    
    try:
        # Lista subdiretórios disponíveis
        abs_base = os.path.abspath(default_base_dir)
        subfolders = []
        
        for root, dirs, files in os.walk(abs_base):
            for d in dirs:
                # Ignora diretórios ocultos e pastas temporárias
                if not d.startswith(".") and d not in ["venv", "node_modules", "__pycache__"]:
                    folder_path = os.path.join(root, d)
                    subfolders.append(folder_path)
        
        # Adiciona o diretório pai como opção
        parent_folder = os.path.dirname(abs_base)
        if parent_folder and parent_folder != abs_base:
            subfolders.insert(0, parent_folder)
        
        # Usa o primeiro diretório encontrado ou o padrão
        return subfolders[0] if subfolders else default_base_dir
    
    except Exception as e:
        st.error(f"Erro ao listar diretórios: {e}")
        return default_base_dir


def initialize_session_state():
    """Inicializa o estado da sessão com valores padrão."""
    if "memoria" not in st.session_state:
        st.session_state["memoria"] = ConversationBufferMemory()
    
    if "folder_path" not in st.session_state:
        st.session_state["folder_path"] = get_folder_path()
    
    if "assistant" not in st.session_state:
        try:
            st.session_state["assistant"] = RAGAssistant()
            st.session_state["initialized"] = True
        except Exception as e:
            st.session_state["error"] = str(e)
            st.session_state["initialized"] = False


def sidebar():
    """Configura e exibe a barra lateral com opções e informações."""
    with st.sidebar:
        st.title("🐕 Pakkun")
        st.subheader("Assistente de Código")
        
        st.markdown("---")
        
        st.markdown(
            """
            ### 📚 Sobre
            
            Pakkun é um assistente especializado em análise de código, 
            utilizando técnicas avançadas de RAG (Retrieval Augmented Generation) 
            para fornecer respostas precisas sobre sua base de código.
            
            ### 🔍 Capacidades
            
            - Responder perguntas sobre a estrutura do código
            - Explicar implementações específicas
            - Analisar relações entre componentes
            - Ajudar com troubleshooting e diagnóstico
            
            ### 🛠️ Base de Dados
            
            Conectado à base local ChromaDB com indexação de:
            - Código-fonte completo
            - Estruturas específicas (classes, funções)
            - Documentação
            """
        )
        
        st.markdown("---")
        
        # Opções avançadas
        with st.expander("⚙️ Opções Avançadas"):
            if st.button("Recarregar Base", type="primary"):
                st.session_state.pop("assistant", None)
                st.session_state.pop("initialized", None)
                st.rerun()
            
            if st.button("Limpar Conversa"):
                st.session_state["memoria"] = ConversationBufferMemory()
                st.rerun()


def chat_interface():
    """Interface principal de chat."""
    st.header("Pakkun - Assistente de Código", divider=True)
    
    # Verifica inicialização
    if not st.session_state.get("initialized", False):
        error = st.session_state.get("error", "Erro desconhecido na inicialização")
        st.error(f"Não foi possível inicializar o assistente: {error}")
        
        if "Base ChromaDB não encontrada" in error:
            st.info(
                "Execute o indexador para criar a base de dados:\n\n"
                "```\npython core/code_indexer.py --folder <caminho_da_pasta>\n```"
            )
        
        return
    
    # Exibe mensagens anteriores
    memoria = st.session_state.get("memoria", ConversationBufferMemory())
    for msg in memoria.buffer_as_messages:
        st.chat_message(msg.type).markdown(msg.content, unsafe_allow_html=True)
    
    # Input do usuário
    if input_user := st.chat_input("Como posso ajudar com seu código?"):
        # Exibe mensagem do usuário
        st.chat_message("human").markdown(input_user)
        
        # Processa a consulta
        with st.spinner("Analisando seu código e preparando resposta..."):
            start_time = time.time()
            
            # Obtém resposta do assistente
            resposta = st.session_state["assistant"].process_query(
                input_user, 
                chat_history=memoria.buffer_as_messages
            )
            
            # Extrai pensamento e resposta
            internal_thought, final_response = extract_thought_and_response(resposta)
            
            # Formata mensagem combinada
            combined_message = ""
            if internal_thought:
                combined_message += (
                    f"<details>\n"
                    f"<summary>Raciocínio detalhado</summary>\n\n"
                    f"{internal_thought}\n"
                    f"</details>\n\n"
                )
            combined_message += final_response
            
            # Calcula tempo de resposta
            elapsed_time = time.time() - start_time
            
            # Exibe resposta
            ai_message = st.chat_message("ai")
            ai_message.markdown(combined_message, unsafe_allow_html=True)
            
            # Adiciona info de tempo de resposta (apenas para debug/demonstração)
            with ai_message:
                st.caption(f"⏱️ Respondido em {elapsed_time:.2f} segundos")
        
        # Atualiza memória
        memoria.chat_memory.add_user_message(input_user)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state["memoria"] = memoria


def main():
    """Função principal do aplicativo."""
    # Configura página
    configure_page()
    
    # Inicializa estado da sessão
    initialize_session_state()
    
    # Exibe barra lateral
    sidebar()
    
    # Exibe interface de chat
    chat_interface()


if __name__ == "__main__":
    main()
