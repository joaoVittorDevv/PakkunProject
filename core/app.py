import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass

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
COLLECTION_MAP_NAME = "collection_map"
COLLECTION_NAME_PREFIX = "file_"
LLM_MODEL = "deepseek-r1-distill-llama-70b"
MAX_CONTEXT_LENGTH = (
    4000  # Aumentando o número máximo de caracteres por documento no contexto
)
MAX_CONTEXT_DOCS = 10  # Aumentando o número máximo de documentos no contexto
DEBUG_MODE = False  # Modo debug para mostrar informações adicionais

# Cores e temas - usando cores padrão do Streamlit para melhor visibilidade
THEME_COLORS = {
    "primary": "#ff4b4b",  # Vermelho Streamlit
    "secondary": "#0068c9",  # Azul Streamlit
    "background": "#4c4c4c",  # Fundo branco
    "text": "#262730",  # Texto escuro padrão
    "code_bg": "#4c4c4c",  # Fundo de código claro
    "success": "#09ab3b",  # Verde Streamlit
    "warning": "#ffbd45",  # Amarelo/laranja Streamlit
    "error": "#ff4b4b",  # Vermelho Streamlit
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
            white-space: pre-wrap;       /* preserve whitespace but wrap text */
            word-wrap: break-word;       /* break long words */
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
        .language-python .hll { background-color: #ffffcc }
        .language-python .c { color: #408080; font-style: italic }
        .language-python .k { color: #008000; font-weight: bold }
        .language-python .o { color: #666666 }
        .language-python .cm { color: #408080; font-style: italic }
        .language-python .cp { color: #BC7A00 }
        .language-python .s { color: #BA2121 }
        .language-python .kc { color: #008000; font-weight: bold }
        .language-python .kd { color: #008000; font-weight: bold }
        .language-python .kt { color: #B00040 }
        .language-python .m { color: #666666 }
        .language-python .s2 { color: #BA2121 }
        .language-javascript .kw { color: #0000FF }
        .language-javascript .str { color: #A31515 }
        .language-javascript .com { color: #008000 }
        .language-javascript .typ { color: #267F99 }
        .language-javascript .lit { color: #36acaa }
        </style>
        """,
        unsafe_allow_html=True,
    )


@dataclass
class FileCollection:
    """Representa uma coleção de arquivo com seus metadados."""

    file_path: str
    collection_name: str
    file_hash: str
    language: str
    file_type: str
    num_chunks: int


class MultiCollectionRAG:
    """
    Implementa RAG avançado com múltiplas coleções ChromaDB por arquivo.

    Características:
    - Carrega o mapeamento de coleções para encontrar arquivos e coleções
    - Implementa estratégias de recuperação específicas por arquivo
    - Permite busca direcionada em arquivos específicos mencionados na consulta
    - Combina resultados de múltiplas coleções com técnicas de reranking
    """

    def __init__(
        self,
        embeddings_model: str = EMBEDDINGS_MODEL,
        persist_dir: str = PERSIST_DIR,
        collection_map_name: str = COLLECTION_MAP_NAME,
        device: Optional[str] = None,
        max_context_docs: int = MAX_CONTEXT_DOCS,
    ):
        """Inicializa o sistema RAG de múltiplas coleções."""
        self.embeddings_model = embeddings_model
        self.persist_dir = persist_dir
        self.collection_map_name = collection_map_name
        self.max_context_docs = max_context_docs

        # Determina o device para embeddings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Inicializa embeddings
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

        # Carrega mapeamento de coleções
        self.collection_map = self._load_collection_map()
        self.file_collections = self._initialize_file_collections()

        # Métricas e estatísticas
        self.collection_stats = {
            "total_files": len(self.file_collections),
            "total_chunks": sum(fc.num_chunks for fc in self.file_collections.values()),
            "languages": self._count_languages(),
            "file_types": self._count_file_types(),
        }

    def _load_collection_map(self) -> Dict[str, Dict[str, Any]]:
        """Carrega o mapeamento de coleções do arquivo JSON."""
        map_path = os.path.join(self.persist_dir, f"{self.collection_map_name}.json")

        if not os.path.exists(map_path):
            raise FileNotFoundError(
                f"Mapeamento de coleções não encontrado em {map_path}. "
                "Certifique-se de que o indexador foi executado corretamente."
            )

        try:
            with open(map_path, "r") as f:
                data = json.load(f)
                return data.get("files", {})
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar mapeamento de coleções: {e}")

    def _initialize_file_collections(self) -> Dict[str, FileCollection]:
        """Inicializa as coleções de arquivo a partir do mapeamento."""
        file_collections = {}

        for file_path, metadata in self.collection_map.items():
            file_collections[file_path] = FileCollection(
                file_path=file_path,
                collection_name=metadata.get("collection_name", ""),
                file_hash=metadata.get("file_hash", ""),
                language=metadata.get("language", "unknown"),
                file_type=metadata.get("file_type", "unknown"),
                num_chunks=metadata.get("num_chunks", 0),
            )

        return file_collections

    def _count_languages(self) -> Dict[str, int]:
        """Conta o número de arquivos por linguagem."""
        languages = {}
        for fc in self.file_collections.values():
            if fc.language not in languages:
                languages[fc.language] = 0
            languages[fc.language] += 1
        return languages

    def _count_file_types(self) -> Dict[str, int]:
        """Conta o número de arquivos por tipo."""
        file_types = {}
        for fc in self.file_collections.values():
            if fc.file_type not in file_types:
                file_types[fc.file_type] = 0
            file_types[fc.file_type] += 1
        return file_types

    def get_collection_for_file(self, file_path: str) -> Optional[Chroma]:
        """Obtém a coleção ChromaDB para um arquivo específico."""
        if file_path not in self.file_collections:
            return None

        fc = self.file_collections[file_path]

        # Cria o cliente Chroma para esta coleção
        chroma = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=fc.collection_name,
        )

        return chroma

    def search_in_specific_files(
        self, query: str, file_paths: List[str], top_k: int = 3
    ) -> List[Document]:
        """
        Busca um query em arquivos específicos.
        Retorna uma lista combinada de documentos relevantes.
        """
        all_results = []

        for file_path in file_paths:
            # Verifica se o arquivo existe no mapeamento
            if file_path not in self.file_collections:
                continue

            # Obtém a coleção para este arquivo
            chroma = self.get_collection_for_file(file_path)
            if not chroma:
                continue

            # Realiza a busca nesta coleção
            retriever = chroma.as_retriever(
                search_type="similarity", search_kwargs={"k": top_k}
            )

            results = retriever.get_relevant_documents(query)
            all_results.extend(results)

        # Ordena por relevância (assumindo que os primeiros resultados de cada coleção são mais relevantes)
        return all_results

    def global_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Realiza uma busca global em todas as coleções.
        Implementação mais complexa que busca em coleções individuais e combina resultados.
        """
        # Implementação atual: busca em coleções individuais e combina resultados
        # Podemos otimizar isso no futuro com índices globais ou técnicas mais avançadas

        # 1. Extrai palavras-chave da consulta para priorizar arquivos relevantes
        keywords = self._extract_keywords(query)

        # 2. Calcula pontuações para cada arquivo com base nas palavras-chave
        file_scores = self._score_files_by_keywords(keywords)

        # 3. Seleciona os N arquivos mais relevantes para busca
        num_files_to_search = min(10, len(self.file_collections))
        top_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[
            :num_files_to_search
        ]

        # 4. Busca nas coleções dos arquivos mais relevantes
        all_results = []
        for file_path, score in top_files:
            chroma = self.get_collection_for_file(file_path)
            if not chroma:
                continue

            # Ajusta o número de resultados com base na pontuação do arquivo
            k = max(1, int(top_k * (score / max(file_scores.values()))))

            retriever = chroma.as_retriever(
                search_type="similarity", search_kwargs={"k": k}
            )

            results = retriever.get_relevant_documents(query)
            all_results.extend(results)

        # 5. Se não encontrou resultados suficientes, expande a busca
        if len(all_results) < top_k:
            # Busca em mais alguns arquivos aleatórios
            import random

            remaining_files = [
                f
                for f in self.file_collections.keys()
                if f not in [path for path, _ in top_files]
            ]
            if remaining_files:
                random_files = random.sample(
                    remaining_files, min(5, len(remaining_files))
                )
                for file_path in random_files:
                    chroma = self.get_collection_for_file(file_path)
                    if not chroma:
                        continue

                    retriever = chroma.as_retriever(
                        search_type="similarity", search_kwargs={"k": 2}
                    )

                    results = retriever.get_relevant_documents(query)
                    all_results.extend(results)

        # 6. Reordena os resultados por similaridade com a consulta
        return self._rerank_results(query, all_results, top_k)

    def _extract_keywords(self, query: str) -> List[str]:
        """Extrai palavras-chave da consulta."""
        # Remove palavras comuns e mantém substantivos, verbos e adjetivos importantes
        # Esta é uma implementação simplificada
        common_words = {
            "o",
            "a",
            "os",
            "as",
            "um",
            "uma",
            "uns",
            "umas",
            "de",
            "da",
            "do",
            "das",
            "dos",
            "em",
            "na",
            "no",
            "nas",
            "nos",
            "por",
            "pela",
            "pelo",
            "pelas",
            "pelos",
            "que",
            "qual",
            "quem",
            "como",
            "onde",
            "quando",
            "por que",
            "porque",
            "é",
            "são",
            "está",
            "estão",
            "foi",
            "foram",
            "será",
            "serão",
            "e",
            "ou",
            "mas",
            "porém",
            "contudo",
            "todavia",
            "entretanto",
            "no entanto",
            "para",
            "com",
            "sem",
            "sobre",
            "sob",
            "entre",
            "após",
            "antes",
            "depois",
        }

        words = re.findall(r"\b\w+\b", query.lower())
        keywords = [
            word for word in words if word not in common_words and len(word) > 2
        ]

        return keywords

    def _score_files_by_keywords(self, keywords: List[str]) -> Dict[str, float]:
        """Pontua arquivos com base em palavras-chave da consulta."""
        scores = {file_path: 0.0 for file_path in self.file_collections.keys()}

        for file_path in scores.keys():
            # Pontua com base no caminho do arquivo
            path_parts = file_path.lower().split("/")
            for keyword in keywords:
                # Pontuação para o nome do arquivo
                if keyword in path_parts[-1]:
                    scores[file_path] += 2.0

                # Pontuação para o caminho do arquivo
                if any(keyword in part for part in path_parts[:-1]):
                    scores[file_path] += 1.0

            # Bônus para arquivos Python e JavaScript (mais comuns em projetos)
            if file_path.endswith(".py"):
                scores[file_path] += 0.5
            elif (
                file_path.endswith(".js")
                or file_path.endswith(".jsx")
                or file_path.endswith(".ts")
            ):
                scores[file_path] += 0.3

        return scores

    def _rerank_results(
        self, query: str, results: List[Document], top_k: int
    ) -> List[Document]:
        """
        Reordena os resultados por relevância.
        Usa uma combinação de similaridade semântica e diversidade.
        """
        if not results:
            return []

        # Deduplicação por conteúdo
        unique_results = []
        seen_content = set()

        for doc in results:
            # Simplifica o conteúdo para comparação (primeiros 100 caracteres)
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)

        # Organiza os resultados por tipo para garantir diversidade
        full_docs = [
            doc for doc in unique_results if not doc.metadata.get("is_chunk", False)
        ]
        chunks = [doc for doc in unique_results if doc.metadata.get("is_chunk", False)]
        code_structures = [
            doc
            for doc in unique_results
            if doc.metadata.get("is_code_structure", False)
        ]

        # Prioriza documento completo, depois estruturas de código e por último os chunks
        reranked = []
        reranked.extend(full_docs[:2])  # Até 2 documentos completos
        reranked.extend(code_structures[:3])  # Até 3 estruturas de código
        reranked.extend(chunks[: top_k - len(reranked)])  # Restante para chunks

        # Se ainda não tiver resultados suficientes, adiciona mais documentos
        remaining = top_k - len(reranked)
        if remaining > 0:
            remaining_docs = [doc for doc in unique_results if doc not in reranked]
            reranked.extend(remaining_docs[:remaining])

        return reranked[:top_k]

    def search_all_files(self) -> List[str]:
        """Retorna a lista de todos os arquivos indexados."""
        return list(self.file_collections.keys())

    def search_by_extension(self, extension: str) -> List[str]:
        """Busca arquivos por extensão."""
        return [
            file_path
            for file_path in self.file_collections.keys()
            if file_path.endswith(extension)
        ]

    def search_by_language(self, language: str) -> List[str]:
        """Busca arquivos por linguagem."""
        return [
            fc.file_path
            for fc in self.file_collections.values()
            if fc.language.lower() == language.lower()
        ]


class RAGAssistant:
    """
    Implementa um assistente de código baseado em técnicas RAG (Retrieval Augmented Generation).

    Funcionalidades:
    - Usa MultiCollectionRAG para busca em coleções individuais por arquivo
    - Filtragem contextual de documentos relevantes
    - Extração de metadados de arquivos mencionados na consulta
    - Reranking de resultados para melhorar relevância
    - Histórico de conversa com memória
    """

    def __init__(
        self,
        embeddings_model: str = EMBEDDINGS_MODEL,
        persist_dir: str = PERSIST_DIR,
        collection_map_name: str = COLLECTION_MAP_NAME,
        llm_model: str = LLM_MODEL,
        device: Optional[str] = None,
        max_context_docs: int = MAX_CONTEXT_DOCS,
    ):
        """Inicializa o assistente RAG."""
        self.embeddings_model = embeddings_model
        self.persist_dir = persist_dir
        self.collection_map_name = collection_map_name
        self.llm_model = llm_model
        self.max_context_docs = max_context_docs

        # Determina o device para embeddings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Inicializa componentes
        self.initialize_components()

        # Configura tracer para monitoramento
        self.tracer = LangChainTracer(
            project_name="PakkunMonitor", tags=["production", "code-analysis", "rag"]
        )

    def initialize_components(self):
        """Inicializa os componentes necessários do sistema RAG."""
        # Inicializa o sistema RAG de múltiplas coleções
        self.multi_collection_rag = MultiCollectionRAG(
            embeddings_model=self.embeddings_model,
            persist_dir=self.persist_dir,
            collection_map_name=self.collection_map_name,
            device=self.device,
            max_context_docs=self.max_context_docs,
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

Use os contextos fornecidos para responder às perguntas:

{context}

Regras obrigatórias:
1. Priorize informações de arquivos explicitamente mencionados na pergunta.
2. Cite as fontes usando o formato: `Fonte: caminho/do/arquivo.ext`.
3. Quando perguntarem sobre existência de arquivos ou módulos, responda com base nas informações completas do sistema.
4. Se um arquivo ou módulo não existir, responda com clareza que "não existe" ou "não está disponível", indicando quais alternativas existem.
5. Para perguntas sobre existência de arquivos, confie nas informações fornecidas no contexto de listagem de arquivos.
6. Seja técnico, preciso e detalhado nas suas explicações de código.
7. Use formatação markdown para destacar trechos de código e conceitos importantes.
8. Se precisar de raciocínio extenso, coloque-o entre tags <think>...</think> - esse conteúdo ficará recolhido na interface.
9. Quando o usuário solicitar informações sobre um componente específico (Django, modelos, views, controllers), busque no collection_map.json para encontrar os arquivos relevantes.

Você pode usar <think>...</think> para mostrar seu raciocínio detalhado antes da resposta final.

Formatos de resposta:
- Para explicações de código: use blocos de código com syntax highlighting
- Para listar arquivos/funções: use listas com marcadores
- Para mostrar relações: descreva a hierarquia e dependências
- Para responder sobre existência de módulos: seja direto e objetivo, fornecendo a informação solicitada

IMPORTANTE: Quando um usuário pergunta se você "tem acesso" ou "conhece" um arquivo ou módulo específico, responda baseado na lista completa de arquivos disponíveis no sistema, não apenas nos documentos recuperados para esta consulta.
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
                query, mentioned_files=mentioned_files
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

    def _find_specific_system_files(self, query: str) -> List[str]:
        """
        Analisa a consulta para identificar referências a tipos de arquivos específicos
        ou componentes do sistema e retorna os arquivos correspondentes do collection_map.json.
        """
        # Lista para armazenar os arquivos encontrados
        found_files = []

        # Carrega o mapeamento de coleções (collection_map.json)
        map_path = os.path.join(self.persist_dir, f"{self.collection_map_name}.json")
        try:
            with open(map_path, "r") as f:
                collection_data = json.load(f)
                files_map = collection_data.get("files", {})
        except Exception as e:
            return []

        # Padrões para identificar referências a tipos de arquivos, módulos ou componentes específicos
        system_patterns = [
            r"\bdjango\b",
            r"\bflask\b",
            r"\bfastapi\b",
            r"\breact\b",
            r"\bangular\b",
            r"\bvue\b",
            r"\bmodelo\b",
            r"\bmodule\b",
            r"\bviews\b",
            r"\burls\b",
            r"\bmodels\b",
            r"\badmin\b",
            r"\bforms\b",
            r"\bserializers\b",
            r"\bviews?\b",
            r"\bcontrollers?\b",
            r"\bcomponents?\b",
            r"\bservices?\b",
            r"\brouters?\b",
            r"\bmiddleware\b",
            r"\bconfiguration\b",
            r"\bsettings\b",
            r"\bdatabase\b",
            r"\bapi\b",
            r"\bendpoint\b",
            r"\bsql\b",
            r"\bmongo\b",
            r"\bprisma\b",
            r"\bmigrations?\b",
            r"\bauth\b",
            r"\bautenticação\b",
            r"\blogin\b",
            r"\bpython\b",
            r"\bjavascript\b",
            r"\btypescript\b",
            r"\bhtml\b",
            r"\bcss\b",
        ]

        # Verifica se a consulta contém algum dos padrões
        matched_patterns = []
        for pattern in system_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                matched_patterns.append(pattern.strip(r"\b"))

        if not matched_patterns:
            return []

        # Para cada padrão encontrado, procura arquivos correspondentes
        for pattern in matched_patterns:
            pattern_lower = pattern.lower()

            # Busca no path dos arquivos
            for file_path, metadata in files_map.items():
                # Verifica no caminho do arquivo
                if pattern_lower in file_path.lower():
                    found_files.append(file_path)
                    continue

                # Verifica no tipo e linguagem do arquivo
                language = metadata.get("language", "").lower()
                file_type = metadata.get("file_type", "").lower()

                if pattern_lower in language or pattern_lower in file_type:
                    found_files.append(file_path)

        # Remove duplicatas
        return list(set(found_files))

    def extract_file_mentions(self, query: str) -> List[str]:
        """
        Extrai menções a arquivos na consulta do usuário.
        Suporta vários formatos de caminhos de arquivo e faz correspondência parcial.
        """
        # Regex aprimorada para capturar diversos formatos de caminhos de arquivo
        file_patterns = [
            r"[\w\-.\\/]+\.(?:py|js|jsx|ts|tsx|md|html|css|java|c|cpp|h|go|rs|php|rb|json|yml|yaml|txt|sh)",  # Caminhos de arquivo
            r'["\'](?:[\w\-.\\/]+\.(?:py|js|jsx|ts|tsx|md|html|css|java|c|cpp|h|go|rs|php|rb|json|yml|yaml|txt|sh))["\']',  # Caminhos entre aspas
        ]

        # Extrai menções diretas a arquivos
        mentioned_files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                # Remove aspas se presentes
                clean_match = match.strip("'\"")
                mentioned_files.append(clean_match)

        # Busca correspondências parciais nos arquivos indexados
        if not mentioned_files:
            # Extrai possíveis nomes de diretórios ou módulos (sem extensão)
            dir_patterns = [
                r"\b[\w\-\.\/\\]+\b",  # Padrão genérico para caminhos
                r'["\'][\w\-\.\/\\]+["\']',  # Caminhos entre aspas
            ]

            potential_dirs = []
            for pattern in dir_patterns:
                matches = re.findall(pattern, query)
                for match in matches:
                    clean_match = match.strip("'\"")
                    if len(clean_match) > 2 and "/" in clean_match:
                        potential_dirs.append(clean_match)

            # Filtra palavras comuns
            common_words = {
                "o",
                "a",
                "os",
                "as",
                "um",
                "uma",
                "uns",
                "umas",
                "de",
                "da",
                "do",
                "das",
                "dos",
                "em",
                "na",
                "no",
                "nas",
                "nos",
                "por",
                "pela",
                "pelo",
                "pelas",
                "pelos",
                "que",
                "qual",
                "quem",
                "como",
                "onde",
                "quando",
                "por que",
                "porque",
            }
            potential_dirs = [
                d
                for d in potential_dirs
                if not any(w == d.lower() for w in common_words)
            ]

            # Adiciona diretórios potenciais à lista
            if potential_dirs:
                mentioned_files.extend(potential_dirs)

            # Tenta extrair partes de nomes de arquivo
            partial_matches = re.findall(r"\b\w+\.[a-zA-Z]{1,5}\b", query)
            if partial_matches:
                all_files = self.multi_collection_rag.search_all_files()
                for partial in partial_matches:
                    for file_path in all_files:
                        if partial in file_path:
                            mentioned_files.append(file_path)

            # Extrai possíveis nomes de modelos de IA
            model_patterns = [
                r"\b(?:gpt-|llama-|mistral-|gemma-|falcon-|deepseek-|phi-|mixtral-|claude-|palm-|vicuna-|qwen-|yi-|xglm-|bloom-|mt0-|t5-|bart-|flan-|opt-|pythia-|bert-|roberta-|distilbert-|t0pp-|alpaca-|cohere-|command-|gemini-|luminous-|davinci-|curie-|babbage-|ada-|turbo-)[\w\-\.]+\b",
                r"\b[\w\-]+-\d+[bB]\b",  # Modelos com tamanho em bilhões de parâmetros (e.g. llama-70b)
                r"\b(?:gpt4?|llama\d|gemma\d|falcon\d|claude\d|phi\d|vicuna\d|mistral\d|qwen\d)\b",
            ]

            for pattern in model_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                mentioned_files.extend(matches)

        # Remove possíveis duplicatas
        return list(set(mentioned_files))

    def multi_strategy_retrieval(
        self,
        query: str,
        mentioned_files: Optional[List[str]] = None,
        k_file_specific: int = 3,
        k_global_search: int = 5,
    ) -> List[Document]:
        """
        Implementa múltiplas estratégias de recuperação para melhorar a qualidade dos resultados.

        1. Verifica se a consulta é sobre listagem/existência de arquivos
        2. Verifica se a consulta está relacionada a um módulo, diretório ou tipo de arquivo específico
        3. Busca em arquivos mencionados explicitamente
        4. Busca global em todas as coleções
        5. Combinação e deduplicação dos resultados
        """
        results = []

        # 1. Verifica se a consulta está relacionada à listagem/existência de arquivos
        # Colocamos isso primeiro para garantir que consultas sobre existência de arquivos ou módulos
        # tenham prioridade e sejam respondidas com a lista completa
        is_file_query = self._is_file_listing_query(query)
        if is_file_query:
            file_listing_doc = self._create_file_listing_document(query)
            if file_listing_doc:
                results.append(file_listing_doc)

                # Se for uma consulta sobre módulo específico, ainda queremos resultados adicionais
                if file_listing_doc.metadata.get(
                    "entity_type"
                ) != "module_check" or not file_listing_doc.metadata.get(
                    "module_exists", False
                ):
                    # Se não tem o módulo ou não é sobre módulo específico, retornamos apenas esta informação
                    if len(results) > 0:
                        return results

        # 2. Verifica se a consulta busca um tipo específico de arquivo, módulo ou componente do sistema
        collection_files = self._find_specific_system_files(query)
        if collection_files:
            # Se encontramos arquivos específicos relacionados à consulta, buscamos neles primeiro
            specific_results = self.multi_collection_rag.search_in_specific_files(
                query, collection_files, top_k=min(5, len(collection_files))
            )
            if specific_results:
                results.extend(specific_results)

        # 3. Busca em arquivos mencionados explicitamente
        file_specific_results = []
        if mentioned_files:
            file_specific_results = self.multi_collection_rag.search_in_specific_files(
                query, mentioned_files, top_k=k_file_specific
            )
            results.extend(file_specific_results)

        # 4. Busca global para complementar
        # Se não encontrou resultados suficientes em arquivos mencionados
        if len(results) < self.max_context_docs:
            # Ajusta o número de resultados globais com base nos já obtidos
            k_remaining = min(k_global_search, self.max_context_docs - len(results))

            if k_remaining > 0:
                global_results = self.multi_collection_rag.global_search(
                    query, top_k=k_remaining
                )
                results.extend(global_results)

        # 5. Se mesmo após a busca não tivermos resultados, mas for uma consulta sobre arquivos,
        # adicionamos as informações de arquivos disponíveis
        if len(results) == 0 or (
            is_file_query
            and all(doc.metadata.get("is_file_listing", False) for doc in results)
        ):
            file_listing_doc = self._create_file_listing_document(query)
            if file_listing_doc and not any(
                doc.metadata.get("is_file_listing", False) for doc in results
            ):
                results.append(file_listing_doc)

        # 6. Deduplica os resultados
        return self.deduplicate_results(results)

    def _is_file_listing_query(self, query: str) -> bool:
        """Verifica se a consulta está relacionada à listagem de arquivos."""
        listing_patterns = [
            r"(?:lista|listar|mostrar?|quais|quais são|que|ver) (?:todos )?(?:os )(?:arquivos|documentos|ficheiros|códigos)",
            r"(?:lista|listar|mostrar?|quais|quais são|que|ver) (?:todos )?(?:os )(?:arquivos|documentos|ficheiros|códigos) (?:que você tem|que você possui|que você conhece|disponíveis)",
            r"(?:quais|que) (?:arquivos|códigos|documentos) (?:você tem|você conhece|estão indexados|estão disponíveis)",
            r"(?:você tem acesso|pode acessar|consegue ver|tem|possui) (?:a|ao|aos|que|quais|o|um) (?:arquivos|códigos|arquivo|módulo|biblioteca|diretório|pasta)",
            r"(?:onde|em que arquivos) está o código",
            r"(?:mostre|liste|mostra|lista) os arquivos (?:de|em|com) (?:python|javascript|typescript|js|ts|py)",
            r"(?:existe|há|tem|contém) (?:algum|um|o|a) (?:arquivo|módulo|classe|componente|biblioteca) (?:chamad[ao]|com o nome|nome de) ([a-zA-Z0-9_\-/]+)",
        ]

        # Verificação direta por padrões conhecidos
        for pattern in listing_patterns:
            if re.search(pattern, query.lower()):
                return True

        # Verificação de referência a nomes específicos que podem não existir
        # Verifica se menciona um nome de módulo ou arquivo específico em contexto de pergunta sobre existência
        existence_words = [
            "acesso",
            "existe",
            "encontrar",
            "disponível",
            "tem",
            "contém",
            "conhece",
            "usa",
            "utilizando",
            "implementa",
        ]
        model_words = [
            "modelo",
            "model",
            "llm",
            "linguagem",
            "language",
            "ia",
            "ai",
            "inteligência",
            "intelligence",
        ]

        # Verifica se está perguntando sobre modelos específicos
        model_pattern = r"\b(?:gpt-|llama-|mistral-|gemma-|falcon-|deepseek-|phi-|mixtral-|claude-|palm-|vicuna-|qwen-|yi-|xglm-|bloom-|mt0-|t5-|bart-|flan-|opt-|pythia-|bert-|roberta-|distilbert-|t0pp-|alpaca-|cohere-|command-|gemini-|luminous-|davinci-|curie-|babbage-|ada-|turbo-)[\w\-\.]+\b"
        if re.search(model_pattern, query, re.IGNORECASE) and any(
            word in query.lower() for word in model_words
        ):
            return True

        # Verificação para diretórios específicos
        if "/" in query and any(word in query.lower() for word in existence_words):
            # Se há menção a um caminho com barras, provavelmente é uma consulta sobre diretório
            return True

        if any(word in query.lower() for word in existence_words):
            # Extrai possíveis nomes de módulos/arquivos da consulta
            module_pattern = r"\b([a-zA-Z][a-zA-Z0-9_\-/]+)\b"
            potential_modules = re.findall(module_pattern, query)

            # Filtra nomes comuns e palavras da pergunta
            common_words = {
                "um",
                "uma",
                "você",
                "como",
                "para",
                "com",
                "por",
                "este",
                "esta",
                "estes",
                "estas",
                "aquele",
                "aquela",
                "aqueles",
                "aquelas",
                "esse",
                "essa",
                "módulo",
                "arquivo",
                "chamado",
                "em",
                "o",
                "a",
                "os",
                "as",
                "do",
                "da",
                "dos",
                "das",
            }
            potential_modules = [
                m for m in potential_modules if m.lower() not in common_words
            ]

            # Se restaram potenciais nomes de módulos, provavelmente é uma consulta de existência
            if potential_modules:
                return True

        return False

    def _create_file_listing_document(self, query: str) -> Optional[Document]:
        """
        Cria um documento com informações sobre arquivos disponíveis.
        Se a consulta for sobre um módulo específico, verifica sua existência.
        """
        # Determina o tipo de arquivo que está sendo solicitado
        file_types = {
            "python": [".py"],
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "markdown": [".md"],
            "html": [".html", ".htm"],
            "css": [".css"],
            "json": [".json"],
            "yaml": [".yml", ".yaml"],
        }

        # Verifica se está perguntando sobre um módulo/arquivo específico
        all_files = self.multi_collection_rag.search_all_files()
        checking_specific_module = False
        specific_module = None

        # Extrai potenciais nomes de módulos da consulta
        module_pattern = r"\b([a-zA-Z][a-zA-Z0-9_]+)\b"
        potential_modules = re.findall(module_pattern, query)

        # Filtra palavras comuns
        common_words = {
            "um",
            "uma",
            "você",
            "como",
            "para",
            "com",
            "por",
            "este",
            "esta",
            "estes",
            "estas",
            "aquele",
            "aquela",
            "aqueles",
            "aquelas",
            "esse",
            "essa",
            "módulo",
            "arquivo",
            "chamado",
            "acesso",
            "existe",
            "tem",
            "possui",
            "conhece",
            "disponível",
            "quais",
            "algum",
            "tem",
        }
        potential_modules = [
            m for m in potential_modules if m.lower() not in common_words
        ]

        # Verifica se algum dos módulos potenciais aparece nos arquivos
        if potential_modules:
            for module in potential_modules:
                # Verifica se existe um arquivo/pasta com esse nome
                module_exists = False
                module_files = []

                # Verifica se o módulo é um nome de modelo de IA
                model_pattern = r"(?:gpt-|llama-|mistral-|gemma-|falcon-|deepseek-|phi-|mixtral-|claude-|palm-|vicuna-|qwen-|yi-|xglm-|bloom-|mt0-|t5-|bart-|flan-|opt-|pythia-|bert-|roberta-|distilbert-|t0pp-|alpaca-|cohere-|command-|gemini-|luminous-|davinci-|curie-|babbage-|ada-|turbo-)[\w\-\.]+"
                if re.match(model_pattern, module, re.IGNORECASE):
                    # Para modelos, verificamos se é o mesmo que estamos usando ou uma variante
                    if (
                        module.lower() == LLM_MODEL.lower()
                        or module.lower() in LLM_MODEL.lower()
                        or LLM_MODEL.lower() in module.lower()
                    ):
                        module_exists = True
                        module_files.append(f"Modelo de IA: {LLM_MODEL} (em uso)")
                    else:
                        # Verifica alguns modelos conhecidos
                        known_models = [
                            "deepseek-r1-distill-llama-70b",
                            "llama-2-70b",
                            "llama-3-70b",
                            "llama-3-8b",
                            "gemma-7b",
                            "gemma-2b",
                            "mistral-7b",
                            "mixtral-8x7b",
                            "claude-3-opus",
                            "claude-3-sonnet",
                            "claude-3-haiku",
                            "gpt-4",
                            "gpt-3.5-turbo",
                        ]

                        if any(
                            model.lower() in module.lower() for model in known_models
                        ):
                            for model in known_models:
                                if model.lower() in module.lower():
                                    module_exists = True
                                    if model == LLM_MODEL:
                                        module_files.append(
                                            f"Modelo de IA: {model} (em uso)"
                                        )
                                    else:
                                        module_files.append(
                                            f"Modelo de IA: {model} (disponível)"
                                        )

                    # Adiciona notas sobre o modelo atual sendo usado
                    if not module_exists:
                        module_files.append(f"Modelo atual em uso: {LLM_MODEL}")
                    continue

                # Primeiro, verifica se estamos buscando um caminho completo
                if "/" in module or "\\" in module:
                    normalized_module = module.replace("\\", "/").lower()
                    # Para consultas de caminho completo, verificamos se qualquer arquivo começa com esse caminho
                    path_prefix = normalized_module.rstrip("/") + "/"

                    for file_path in all_files:
                        normalized_path = file_path.replace("\\", "/").lower()

                        # Correspondência exata do caminho
                        if normalized_path == normalized_module:
                            module_exists = True
                            module_files.append(file_path)

                        # Correspondência de pasta/prefixo
                        elif normalized_path.startswith(path_prefix):
                            module_exists = True
                            module_files.append(file_path)

                        # Se caminho for parte de um arquivo
                        elif normalized_module in normalized_path:
                            # Verifica se a correspondência respeita a estrutura do diretório
                            path_parts = normalized_module.split("/")
                            file_parts = normalized_path.split("/")

                            for i in range(len(file_parts) - len(path_parts) + 1):
                                if i < len(file_parts) and i + len(path_parts) <= len(
                                    file_parts
                                ):
                                    if (
                                        file_parts[i : i + len(path_parts)]
                                        == path_parts
                                    ):
                                        module_exists = True
                                        module_files.append(file_path)
                                        break
                else:
                    # Para módulos/pastas simples
                    normalized_module = module.lower()
                    for file_path in all_files:
                        normalized_path = file_path.replace("\\", "/").lower()

                        # Verifica se o módulo é um diretório no caminho
                        if f"/{normalized_module}/" in normalized_path:
                            module_exists = True
                            module_files.append(file_path)

                        # Verifica se o módulo é um arquivo
                        elif (
                            f"/{normalized_module}." in normalized_path
                            or normalized_path.endswith(f"/{normalized_module}")
                        ):
                            module_exists = True
                            module_files.append(file_path)

                        # Verifica nome de arquivo sem extensão
                        elif (
                            os.path.splitext(os.path.basename(normalized_path))[0]
                            == normalized_module
                        ):
                            module_exists = True
                            module_files.append(file_path)

                        # Verifica se o nome do módulo está em alguma parte do caminho
                        elif len(
                            normalized_module
                        ) > 3 and normalized_module in os.path.basename(
                            normalized_path
                        ):
                            module_exists = True
                            module_files.append(file_path)

                # Verifica também cada parte do caminho para encontrar pastas relacionadas
                path_parts = module.split("/")
                if len(path_parts) > 1:
                    # Se é um caminho como "lgpd-10/backend/pessoas"
                    nested_dirs = {}
                    all_dirs = set()

                    # Primeiro, extraímos todos os diretórios do sistema
                    for file_path in all_files:
                        normalized_path = file_path.replace("\\", "/")
                        dir_parts = normalized_path.split("/")
                        current_path = ""

                        for part in dir_parts[:-1]:  # Excluindo o nome do arquivo
                            if current_path:
                                current_path += "/" + part
                            else:
                                current_path = part
                            all_dirs.add(current_path)

                    # Agora, verifica se o caminho solicitado existe
                    if module in all_dirs:
                        module_exists = True
                        # Encontra arquivos nesta pasta
                        for file_path in all_files:
                            if file_path.replace("\\", "/").startswith(module + "/"):
                                module_files.append(file_path)

                    # Se não encontrou o caminho exato, verifica parciais
                    if not module_exists:
                        for file_path in all_files:
                            normalized_path = file_path.replace("\\", "/")
                            for i in range(len(path_parts)):
                                partial_path = "/".join(path_parts[: i + 1])
                                if f"{partial_path}/" in normalized_path:
                                    if partial_path not in nested_dirs:
                                        nested_dirs[partial_path] = []
                                    nested_dirs[partial_path].append(file_path)

                        # Se encontrou diretórios parciais, verifica se temos o caminho específico
                        if nested_dirs:
                            # Verifica se temos o módulo completo
                            if module in nested_dirs:
                                module_exists = True
                                module_files.extend(nested_dirs[module][:20])

                            # Ou pelo menos uma parte relevante do caminho
                            for partial_path, files in nested_dirs.items():
                                if not module_exists and partial_path.startswith(
                                    path_parts[0]
                                ):
                                    # Verifica se temos pelo menos o último componente
                                    if path_parts[-1] in partial_path:
                                        module_exists = True
                                        module_files.extend(files[:10])

                    # Se encontrou o diretório mas não tem arquivos, adiciona info sobre a pasta
                    if module_exists and not module_files:
                        module_files.append(
                            f"Diretório {module} existe, mas não contém arquivos diretos."
                        )
                        # Tenta encontrar subdiretórios
                        subdirs = [
                            dir_path
                            for dir_path in all_dirs
                            if dir_path.startswith(module + "/")
                        ]
                        if subdirs:
                            subdirs = sorted(subdirs)[:5]  # Limita a 5 subdiretórios
                            module_files.append(f"Subdiretórios: {', '.join(subdirs)}")

                if module_exists:
                    checking_specific_module = True
                    specific_module = module

                    # Formata a resposta específica para esse módulo
                    # Remove duplicatas de module_files se houver
                    module_files = list(set(module_files))

                    # Limita a quantidade de arquivos exibidos
                    max_files_to_show = 15
                    files_display = module_files[:max_files_to_show]

                    # Adiciona mensagem sobre arquivos adicionais
                    more_files_msg = ""
                    if len(module_files) > max_files_to_show:
                        more_files_msg = f"\n\n...e mais {len(module_files) - max_files_to_show} arquivo(s) não listados."

                    content = f"""
Informações sobre o módulo/diretório solicitado: "{module}"

SIM, tenho acesso a arquivos relacionados a "{module}".
Foram encontrados {len(module_files)} arquivo(s) relacionados:

{chr(10).join([f"- {file}" for file in files_display])}{more_files_msg}

Você pode fazer perguntas específicas sobre este módulo.
"""
                    return Document(
                        page_content=content,
                        metadata={
                            "source": "file_listing",
                            "entity_type": "module_check",
                            "module_name": module,
                            "module_exists": True,
                            "is_file_listing": True,
                        },
                    )

                # Se o módulo não foi encontrado, mas foi mencionado com palavras de existência
                existence_words = [
                    "acesso",
                    "existe",
                    "encontrar",
                    "disponível",
                    "tem",
                    "contém",
                    "conhece",
                ]
                if any(word in query.lower() for word in existence_words):
                    checking_specific_module = True
                    specific_module = module

                    # Tenta encontrar parciais do caminho para sugerir alternativas
                    path_parts = module.split("/")
                    suggested_dirs = set()

                    # Se é uma consulta por um caminho aninhado
                    if len(path_parts) > 1:
                        # Verifica quais são os diretórios que contêm o começo do caminho
                        partial_prefix = "/".join(
                            path_parts[:-1]
                        )  # ex: "lgpd-10/backend"
                        last_part = path_parts[-1]  # ex: "pessoas"

                        all_dirs = set()
                        similar_dirs = set()

                        # Extrai todos os diretórios do sistema
                        for file_path in all_files:
                            normalized_path = file_path.replace("\\", "/")
                            dir_parts = normalized_path.split("/")
                            current_path = ""

                            for part in dir_parts[:-1]:  # Excluindo o nome do arquivo
                                if current_path:
                                    current_path += "/" + part
                                else:
                                    current_path = part
                                all_dirs.add(current_path)

                                # Procura por diretórios similares
                                if current_path.startswith(partial_prefix):
                                    similar_dirs.add(current_path)

                        # Adiciona diretorios do mesmo nível que foram encontrados
                        if similar_dirs:
                            for dir_path in sorted(similar_dirs):
                                suggested_dirs.add(dir_path)

                        # Se não encontrou nada no mesmo nível, mostra o nível superior
                        if not suggested_dirs:
                            for dir_path in sorted(all_dirs):
                                if partial_prefix in dir_path:
                                    suggested_dirs.add(dir_path)

                    # Se não conseguiu encontrar sugestões relacionadas, mostra diretórios de alto nível
                    if not suggested_dirs:
                        top_level_dirs = set()
                        for file_path in all_files:
                            parts = file_path.split("/")
                            if len(parts) > 1:
                                top_level_dirs.add(parts[0])

                        for dir_name in sorted(top_level_dirs):
                            if dir_name:
                                suggested_dirs.add(dir_name)

                    top_dirs_str = "\n".join(
                        [f"- {dir}" for dir in sorted(suggested_dirs) if dir]
                    )

                    # Adapta a mensagem com base em se é um caminho aninhado ou não
                    if "/" in module:
                        dirs_msg = "Diretórios relacionados ou similares:"
                    else:
                        dirs_msg = "Os principais diretórios/módulos disponíveis são:"

                    content = f"""
Informações sobre o módulo/diretório solicitado: "{module}"

NÃO, não encontrei nenhum arquivo ou módulo chamado "{module}" nos dados indexados.

{dirs_msg}
{top_dirs_str}

Total de arquivos indexados: {len(all_files)}
"""
                    return Document(
                        page_content=content,
                        metadata={
                            "source": "file_listing",
                            "entity_type": "module_check",
                            "module_name": module,
                            "module_exists": False,
                            "is_file_listing": True,
                        },
                    )

        # Se não estiver verificando um módulo específico, continua com o comportamento normal
        # Verifica se a consulta está buscando um tipo específico de arquivo
        specific_type = None
        for file_type, extensions in file_types.items():
            if file_type in query.lower() or any(
                ext in query.lower() for ext in extensions
            ):
                specific_type = file_type
                break

        # Obtém a lista de arquivos
        if specific_type:
            # Busca arquivos específicos deste tipo
            files = []
            for ext in file_types.get(specific_type, []):
                files.extend(self.multi_collection_rag.search_by_extension(ext))
        else:
            # Busca todos os arquivos
            files = all_files

        # Limita o número de arquivos para não sobrecarregar o contexto
        max_files = 50
        if len(files) > max_files:
            files = files[:max_files]
            file_list_content = "\n".join([f"- {file}" for file in files])
            file_list_content += (
                f"\n\n...e mais {len(files) - max_files} arquivos não listados."
            )
        else:
            file_list_content = "\n".join([f"- {file}" for file in files])

        # Organiza os arquivos por extensão
        extension_count = {}
        for file in all_files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in extension_count:
                extension_count[ext] = 0
            extension_count[ext] += 1

        extension_summary = "\n".join(
            [
                f"- {ext}: {count} arquivo(s)"
                for ext, count in sorted(
                    extension_count.items(), key=lambda x: x[1], reverse=True
                )
            ]
        )

        # Cria o documento
        content = f"""
Informações sobre arquivos disponíveis:

Total de arquivos indexados: {len(all_files)}

Distribuição por extensão:
{extension_summary}

{f'Arquivos {specific_type} disponíveis:' if specific_type else 'Exemplos de arquivos disponíveis:'}
{file_list_content}
"""

        return Document(
            page_content=content,
            metadata={
                "source": "file_listing",
                "entity_type": "file_listing",
                "is_file_listing": True,
            },
        )

    def deduplicate_results(self, results: List[Document]) -> List[Document]:
        """
        Deduplica resultados e prioriza documentos mais relevantes.
        """
        # Remove duplicatas baseadas no conteúdo
        seen_contents = set()
        unique_docs = []

        for doc in results:
            # Usa um hash do conteúdo para verificar duplicação
            content_hash = hash(
                doc.page_content[:100]
            )  # Usa apenas o início para comparação
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)

        # Retorna até o máximo de documentos permitidos
        return unique_docs[: self.max_context_docs]

    def format_context_for_llm(self, docs: List[Document]) -> str:
        """
        Formata documentos recuperados em um formato otimizado para o contexto do LLM.
        Inclui metadados importantes e limita o tamanho para evitar exceder o contexto.
        """
        context_parts = []

        # Adiciona informação sobre o mapeamento de coleções disponíveis
        collection_map_info = self._get_collection_map_summary()
        if collection_map_info:
            context_parts.append(collection_map_info)

        for i, doc in enumerate(docs, 1):
            # Extrai metadados importantes
            source = doc.metadata.get("source", "Unknown")
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

        # Se não houver resultados, adiciona uma mensagem informativa
        if len(context_parts) <= 1:  # Apenas tem o resumo do collection_map
            return "Não foram encontrados documentos relevantes para esta consulta. Responda com as informações gerais que você possui sobre programação, mas indique claramente quando não tiver informações específicas sobre o código mencionado."

        # Junta os documentos formatados
        return "\n---\n".join(context_parts)

    def _get_collection_map_summary(self) -> str:
        """
        Obtém um resumo do mapeamento de coleções para fornecer contexto sobre arquivos disponíveis.
        """
        map_path = os.path.join(self.persist_dir, f"{self.collection_map_name}.json")
        try:
            with open(map_path, "r") as f:
                collection_data = json.load(f)
                files_map = collection_data.get("files", {})

            if not files_map:
                return ""

            # Conta tipos de arquivos disponíveis
            file_types = {}
            languages = {}

            for file_path, metadata in files_map.items():
                file_type = metadata.get("file_type", "unknown")
                language = metadata.get("language", "unknown")

                if file_type not in file_types:
                    file_types[file_type] = 0
                file_types[file_type] += 1

                if language not in languages:
                    languages[language] = 0
                languages[language] += 1

            # Formata o resumo
            summary = "[Collection Map Summary]\n"
            summary += f"Total de arquivos disponíveis: {len(files_map)}\n"

            summary += "\nTipos de arquivo:\n"
            for file_type, count in file_types.items():
                summary += f"- {file_type}: {count}\n"

            summary += "\nLinguagens disponíveis:\n"
            for language, count in languages.items():
                summary += f"- {language}: {count}\n"

            return summary

        except Exception as e:
            return ""

    def process_query(
        self,
        query: str,
        chat_history: Optional[List[Union[HumanMessage, AIMessage]]] = None,
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
            config={"callbacks": [self.tracer]},
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
                if not d.startswith(".") and d not in [
                    "venv",
                    "node_modules",
                    "__pycache__",
                ]:
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
            - Arquivo individual em coleção separada
            - Extratores específicos para Python e JavaScript
            - Estruturas específicas (classes, funções)
            - Documentação com seções extraídas
            """
        )

        st.markdown("---")

        # Estatísticas da base de dados
        if st.session_state.get("initialized", False):
            assistant = st.session_state["assistant"]
            if hasattr(assistant, "multi_collection_rag"):
                stats = assistant.multi_collection_rag.collection_stats

                with st.expander("📊 Estatísticas da Base"):
                    st.write(f"Total de arquivos: {stats['total_files']}")
                    st.write(f"Total de chunks: {stats['total_chunks']}")

                    st.subheader("Linguagens")
                    for lang, count in stats["languages"].items():
                        st.write(f"- {lang}: {count}")

                    st.subheader("Tipos de arquivo")
                    for file_type, count in stats["file_types"].items():
                        st.write(f"- {file_type}: {count}")

        # Opções avançadas
        with st.expander("⚙️ Opções Avançadas"):
            if st.button("Recarregar Base", type="primary"):
                st.session_state.pop("assistant", None)
                st.session_state.pop("initialized", None)
                st.rerun()

            if st.button("Limpar Conversa"):
                st.session_state["memoria"] = ConversationBufferMemory()
                st.rerun()

            # Opção para debug
            debug_mode = st.checkbox("Modo Debug", value=DEBUG_MODE)
            if debug_mode != DEBUG_MODE:
                st.session_state["debug_mode"] = debug_mode
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
                input_user, chat_history=memoria.buffer_as_messages
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

                # Mostra informações de debug se estiver no modo debug
                if st.session_state.get("debug_mode", DEBUG_MODE):
                    with st.expander("🔍 Debug info"):
                        st.write("Arquivos mencionados na consulta:")
                        mentioned_files = st.session_state[
                            "assistant"
                        ].extract_file_mentions(input_user)
                        st.write(mentioned_files or "Nenhum arquivo mencionado")

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
