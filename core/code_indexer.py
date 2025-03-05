import argparse
import hashlib
import json
import logging
import os
import re
from typing import List, Dict, Optional, Tuple, Any, Set

import chardet
import torch
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

# Configurações
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME_PREFIX = "file_"
COLLECTION_MAP_NAME = "collection_map"
CHUNK_SIZE = 2500  # Aumentando tamanho de chunk para mais contexto
CHUNK_OVERLAP = 300  # Aumentando sobreposição para melhor contexto

ALLOWED_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".json", ".yml", ".yaml", 
    ".txt", ".sh", ".dockerfile", ".html", ".css", ".vue", ".java", ".c", 
    ".cpp", ".h", ".cs", ".go", ".php", ".rb", ".rust", ".swift"
}

EXCLUDED_DIRS = {
    "node_modules", "venv", ".venv", "__pycache__", ".git", ".idea", 
    "env", "dist", "build", ".pytest_cache", "__snapshots__", "chroma_db"
}

MAX_FILE_SIZE = 100_000_000  # 100 MB

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


class CodeIndexer:
    """
    Classe responsável por indexar código fonte e documentação.
    Usa ChromaDB para armazenar embeddings, com coleções separadas para cada arquivo.
    """
    def __init__(
        self,
        embeddings_model: str = EMBEDDINGS_MODEL,
        persist_dir: str = PERSIST_DIR,
        collection_prefix: str = COLLECTION_NAME_PREFIX,
        device: Optional[str] = None
    ):
        self.embeddings_model = embeddings_model
        self.persist_dir = persist_dir
        self.collection_prefix = collection_prefix
        self.collection_map_name = COLLECTION_MAP_NAME
        
        # Determina o device para embeddings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Utilizando device para embeddings: {self.device}")
        
        # Inicializa o modelo de embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Inicializa os text splitters específicos por linguagem
        self._init_text_splitters()
        
        # Contadores para estatísticas
        self.stats = {
            "loaded": 0,
            "skipped_ext": 0,
            "skipped_size": 0,
            "error": 0,
            "total_chunks": 0,
            "total_files": 0,
            "total_collections": 0,
        }
        
        # Mapeamento de arquivos para coleções
        self.collection_map = {}
    
    def _init_text_splitters(self):
        """
        Inicializa text splitters específicos para diferentes linguagens.
        """
        # Splitter genérico para uso padrão
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True,
        )
        
        # Splitter específico para Python
        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            keep_separator=True,
        )
        
        # Splitter específico para JavaScript/TypeScript
        self.js_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JS,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            keep_separator=True,
        )
        
        # Mapeamento de extensões para text splitters específicos
        self.language_splitters = {
            "python": self.python_splitter,
            "javascript": self.js_splitter,
            "typescript": self.js_splitter,
            "react": self.js_splitter,
            "react-typescript": self.js_splitter,
        }

    def load_documents_from_folder(self, folder_path: str) -> Dict[str, List[Document]]:
        """
        Carrega documentos de código do diretório especificado.
        Retorna um dicionário mapeando caminhos de arquivo para listas de documentos.
        """
        logger.info(f"Carregando documentos do diretório: {folder_path}")
        docs_by_file = {}

        for root, dirs, files in os.walk(folder_path):
            # Filtra diretórios excluídos
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in ALLOWED_EXTS:
                    self.stats["skipped_ext"] += 1
                    continue

                full_path = os.path.join(root, file)
                try:
                    # Verifica tamanho do arquivo
                    if os.path.getsize(full_path) > MAX_FILE_SIZE:
                        logger.warning(f"Arquivo grande ignorado: {full_path}")
                        self.stats["skipped_size"] += 1
                        continue

                    # Lê o arquivo com detecção automática de encoding
                    with open(full_path, "rb") as f:
                        raw_data = f.read()
                    
                    result = chardet.detect(raw_data)
                    encoding = result.get("encoding") or "utf-8"
                    text = raw_data.decode(encoding, errors="replace").strip()
                    
                    if len(text) < 10:  # Ignora arquivos vazios ou muito pequenos
                        self.stats["skipped_ext"] += 1
                        continue

                    # Caminho relativo para tornar os metadados mais legíveis
                    rel_path = os.path.relpath(full_path, folder_path)
                    logger.debug(f"Processando arquivo: {rel_path}")

                    # Determina tipo de arquivo e linguagem
                    file_type, language = self._get_file_metadata(ext)

                    # Metadados aprimorados
                    metadata = {
                        "source": rel_path,
                        "ext": ext[1:] if ext else "",
                        "full_path": full_path,
                        "file_size": len(text),
                        "file_type": file_type,
                        "language": language,
                        "is_chunk": False,  # Documento completo
                    }

                    # Adiciona o documento ao dicionário
                    docs_by_file[rel_path] = [Document(page_content=text, metadata=metadata)]
                    self.stats["loaded"] += 1

                except Exception as e:
                    logger.error(f"Erro ao carregar {full_path}: {str(e)}")
                    self.stats["error"] += 1

        logger.info(f"Documentos carregados: {self.stats['loaded']}")
        logger.info(f"Documentos ignorados por extensão: {self.stats['skipped_ext']}")
        logger.info(f"Arquivos muito grandes ignorados: {self.stats['skipped_size']}")
        logger.info(f"Erros encontrados: {self.stats['error']}")
        
        return docs_by_file

    def _get_file_metadata(self, ext: str) -> Tuple[str, str]:
        """
        Determina o tipo e linguagem de arquivo com base na extensão.
        """
        code_extensions = {
            ".py": ("code", "python"),
            ".js": ("code", "javascript"),
            ".ts": ("code", "typescript"),
            ".jsx": ("code", "react"),
            ".tsx": ("code", "react-typescript"),
            ".java": ("code", "java"),
            ".c": ("code", "c"),
            ".cpp": ("code", "cpp"),
            ".cs": ("code", "csharp"),
            ".go": ("code", "go"),
            ".rb": ("code", "ruby"),
            ".php": ("code", "php"),
            ".html": ("code", "html"),
            ".css": ("code", "css"),
            ".vue": ("code", "vue"),
        }
        
        document_extensions = {
            ".md": ("document", "markdown"),
            ".txt": ("document", "text"),
            ".json": ("data", "json"),
            ".yml": ("data", "yaml"),
            ".yaml": ("data", "yaml"),
        }
        
        if ext in code_extensions:
            return code_extensions[ext]
        elif ext in document_extensions:
            return document_extensions[ext]
        else:
            return ("unknown", "unknown")

    def process_documents(self, docs_by_file: Dict[str, List[Document]]) -> Dict[str, List[Document]]:
        """
        Processa os documentos para cada arquivo:
        1. Mantém o documento completo
        2. Gera chunks usando o splitter apropriado para a linguagem
        3. Extrai estruturas importantes (classes, funções, etc.)
        
        Retorna um dicionário mapeando caminhos de arquivo para listas de documentos processados.
        """
        processed_docs = {}
        
        logger.info("Processando documentos para indexação...")
        
        for file_path, docs in tqdm(docs_by_file.items(), desc="Processando arquivos"):
            if not docs:
                continue
                
            doc = docs[0]  # Cada arquivo tem um documento principal
            content = doc.page_content
            language = doc.metadata.get("language", "").lower()
            file_type = doc.metadata.get("file_type", "unknown")
            ext = doc.metadata.get("ext", "").lower()
            
            # Cria a lista de documentos processados, começando com o documento completo
            processed_file_docs = [doc]
            
            # Chunks para busca semântica mais granular
            # Escolhe o splitter apropriado para a linguagem
            splitter = self.language_splitters.get(language, self.default_splitter)
            chunks = splitter.split_text(content)
            
            # Adiciona chunks ao resultado
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_id": i,
                    "is_chunk": True,
                    "chunk_size": len(chunk),
                    "entity_type": "text_chunk",
                }
                
                processed_file_docs.append(Document(page_content=chunk, metadata=chunk_metadata))
            
            # Extrai estruturas específicas com base no tipo de arquivo
            if file_type == "code":
                self._extract_code_structures(doc, processed_file_docs)
            
            # Para Markdown, extrai seções
            elif ext == "md":
                self._extract_markdown_sections(doc, processed_file_docs)
            
            # Adiciona os documentos processados ao resultado
            processed_docs[file_path] = processed_file_docs
            self.stats["total_chunks"] += len(processed_file_docs) - 1  # -1 para excluir o documento completo
        
        self.stats["total_files"] = len(processed_docs)
        logger.info(f"Total de arquivos processados: {self.stats['total_files']}")
        logger.info(f"Total de chunks e estruturas extraídas: {self.stats['total_chunks']}")
        
        return processed_docs

    def _extract_code_structures(self, doc: Document, result_docs: List[Document]) -> None:
        """
        Extrai estruturas importantes de código (classes, funções, etc.) e adiciona
        à lista de documentos para indexação.
        """
        ext = doc.metadata.get("ext", "").lower()
        language = doc.metadata.get("language", "").lower()
        content = doc.page_content
        
        # Extração específica por linguagem
        if language == "python":
            # Extração mais complexa para Python
            self._extract_python_structures(doc, content, result_docs)
            
        elif language in ["javascript", "typescript", "react", "react-typescript"]:
            # Extração melhorada para JS/TS
            self._extract_js_structures(doc, content, result_docs)

    def _extract_python_structures(self, doc: Document, content: str, result_docs: List[Document]) -> None:
        """
        Extração refinada de estruturas Python com melhor reconhecimento de contexto.
        """
        # Classes com captura de documentação e métodos
        class_pattern = r"(class\s+\w+(?:\([^)]*\))?\s*:(?:[\s\S]*?)(?=\nclass|\Z))"
        classes = re.finditer(class_pattern, content)
        
        for match in classes:
            class_content = match.group(0)
            # Encontra o nome da classe
            class_name_match = re.search(r"class\s+(\w+)", class_content)
            if class_name_match:
                class_name = class_name_match.group(1)
                
                # Verifica se o tamanho é suficiente
                if len(class_content) >= 100:
                    result_docs.append(
                        Document(
                            page_content=class_content,
                            metadata={
                                **doc.metadata,
                                "entity_type": "class",
                                "entity_name": class_name,
                                "is_chunk": False,
                                "is_code_structure": True,
                            },
                        )
                    )
        
        # Funções e métodos
        func_pattern = r"(def\s+\w+\([^)]*\)\s*(?:->(?:[^:]+))?\s*:(?:[\s\S]*?)(?=\ndef|\nclass|\Z))"
        functions = re.finditer(func_pattern, content)
        
        for match in functions:
            func_content = match.group(0)
            # Encontra o nome da função
            func_name_match = re.search(r"def\s+(\w+)", func_content)
            if func_name_match:
                func_name = func_name_match.group(1)
                
                # Tenta capturar docstring se existir
                docstring_match = re.search(r'"""([\s\S]*?)"""', func_content)
                has_docstring = bool(docstring_match)
                
                # Extrai também o contexto (indentação)
                is_method = re.search(r'^\s+def', func_content, re.MULTILINE) is not None
                
                # Verifica se o tamanho é suficiente
                if len(func_content) >= 50:
                    result_docs.append(
                        Document(
                            page_content=func_content,
                            metadata={
                                **doc.metadata,
                                "entity_type": "method" if is_method else "function",
                                "entity_name": func_name,
                                "has_docstring": has_docstring,
                                "is_chunk": False,
                                "is_code_structure": True,
                            },
                        )
                    )
        
        # Extrai decoradores (@app.route, @property, etc.)
        decorator_pattern = r"(@\w+(?:\.\w+)*(?:\([^)]*\))?\s*\n+\s*def\s+\w+(?:[\s\S]*?)(?=\n\S|\Z))"
        decorators = re.finditer(decorator_pattern, content)
        
        for match in decorators:
            decorator_content = match.group(0)
            # Encontra o nome do decorador
            decorator_name_match = re.search(r"@([\w\.]+)", decorator_content)
            func_name_match = re.search(r"def\s+(\w+)", decorator_content)
            
            if decorator_name_match and func_name_match:
                decorator_name = decorator_name_match.group(1)
                func_name = func_name_match.group(1)
                
                # Verifica se o tamanho é suficiente
                if len(decorator_content) >= 50:
                    result_docs.append(
                        Document(
                            page_content=decorator_content,
                            metadata={
                                **doc.metadata,
                                "entity_type": "decorated_function",
                                "entity_name": func_name,
                                "decorator": decorator_name,
                                "is_chunk": False,
                                "is_code_structure": True,
                            },
                        )
                    )

    def _extract_js_structures(self, doc: Document, content: str, result_docs: List[Document]) -> None:
        """
        Extração refinada de estruturas JavaScript/TypeScript.
        """
        # Extrai classes
        class_pattern = r"(class\s+\w+(?:\s+extends\s+\w+)?\s*\{[\s\S]*?(?=\n\}\n|\}\Z))"
        classes = re.finditer(class_pattern, content)
        
        for match in classes:
            class_content = match.group(0) + "\n}"  # Adiciona o fechamento da classe
            class_name_match = re.search(r"class\s+(\w+)", class_content)
            
            if class_name_match:
                class_name = class_name_match.group(1)
                
                # Verifica se o tamanho é suficiente
                if len(class_content) >= 100:
                    result_docs.append(
                        Document(
                            page_content=class_content,
                            metadata={
                                **doc.metadata,
                                "entity_type": "class",
                                "entity_name": class_name,
                                "is_chunk": False,
                                "is_code_structure": True,
                            },
                        )
                    )
        
        # Componentes React (função ou const)
        component_patterns = [
            # Funções de componente React
            r"(function\s+(\w+)\s*\([^)]*\)\s*\{[\s\S]*?(?=\n\}\n|\}\Z))",
            # Arrow functions para componentes React
            r"(const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{[\s\S]*?(?=\n\}\n|\}\Z))",
            # Arrow functions sem chaves (JSX direto)
            r"(const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\([\s\S]*?(?=\n\)\n|\)\Z))",
            # Exportações nomeadas e default
            r"(export\s+(?:default\s+)?(?:function|const|class)\s+\w+[\s\S]*?(?=\n\}\n|\}\Z|\)\Z))"
        ]
        
        for pattern in component_patterns:
            components = re.finditer(pattern, content)
            for match in components:
                component_content = match.group(1)
                # Adiciona } de fechamento se necessário e não estiver incluído
                if "{" in component_content and not component_content.rstrip().endswith("}"):
                    component_content += "\n}"
                # Adiciona ) de fechamento se necessário e não estiver incluído
                elif "=>" in component_content and "(" in component_content and not component_content.rstrip().endswith(")"):
                    component_content += "\n)"
                
                # Tenta obter o nome do componente
                name_match = re.search(r"(?:function|const|class)\s+(\w+)", component_content)
                if name_match:
                    component_name = name_match.group(1)
                    
                    # Verifica se é um componente React (letra maiúscula ou contém JSX)
                    is_react = (
                        component_name[0].isupper() or 
                        "<" in component_content or 
                        "React" in component_content
                    )
                    
                    entity_type = "react_component" if is_react else "function"
                    
                    # Verifica se o tamanho é suficiente
                    if len(component_content) >= 50:
                        result_docs.append(
                            Document(
                                page_content=component_content,
                                metadata={
                                    **doc.metadata,
                                    "entity_type": entity_type,
                                    "entity_name": component_name,
                                    "is_chunk": False,
                                    "is_code_structure": True,
                                },
                            )
                        )
        
        # Hooks React (useState, useEffect, etc.)
        hook_pattern = r"(const\s+\[\w+,\s*\w+\]\s*=\s*useState[\s\S]*?;)"
        hooks = re.finditer(hook_pattern, content)
        
        for match in hooks:
            hook_content = match.group(1)
            # Tenta obter o nome do estado
            state_match = re.search(r"const\s+\[(\w+),", hook_content)
            if state_match:
                state_name = state_match.group(1)
                
                # Verifica se o tamanho é suficiente
                if len(hook_content) >= 20:
                    result_docs.append(
                        Document(
                            page_content=hook_content,
                            metadata={
                                **doc.metadata,
                                "entity_type": "react_hook",
                                "entity_name": state_name,
                                "is_chunk": False,
                                "is_code_structure": True,
                            },
                        )
                    )

    def _extract_markdown_sections(self, doc: Document, result_docs: List[Document]) -> None:
        """
        Extrai seções de documentos markdown baseadas em cabeçalhos.
        """
        content = doc.page_content
        header_pattern = r"(#+)\s+(.*)"
        lines = content.split("\n")
        current_section = []
        current_header = "Início do documento"
        current_level = 0
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            if header_match:
                # Salva a seção anterior antes de começar uma nova
                if current_section:
                    section_content = "\n".join(current_section)
                    if len(section_content) > 100:
                        result_docs.append(
                            Document(
                                page_content=section_content,
                                metadata={
                                    **doc.metadata,
                                    "entity_type": "markdown_section",
                                    "entity_name": current_header,
                                    "header_level": current_level,
                                    "is_chunk": False,
                                    "is_markdown_section": True,
                                },
                            )
                        )
                
                # Inicia nova seção
                current_header = header_match.group(2)
                current_level = len(header_match.group(1))  # Número de # no cabeçalho
                current_section = [line]
            else:
                current_section.append(line)
        
        # Adiciona a última seção
        if current_section:
            section_content = "\n".join(current_section)
            if len(section_content) > 100:
                result_docs.append(
                    Document(
                        page_content=section_content,
                        metadata={
                            **doc.metadata,
                            "entity_type": "markdown_section",
                            "entity_name": current_header,
                            "header_level": current_level,
                            "is_chunk": False,
                            "is_markdown_section": True,
                        },
                    )
                )

    def create_file_hash(self, file_path: str) -> str:
        """
        Cria um hash para o caminho do arquivo para uso como nome de coleção.
        """
        # Usa MD5 para criar um hash do caminho do arquivo
        hash_obj = hashlib.md5(file_path.encode())
        return hash_obj.hexdigest()

    def index_documents(self, processed_docs: Dict[str, List[Document]]) -> Dict[str, Chroma]:
        """
        Indexa os documentos no ChromaDB, criando uma coleção separada para cada arquivo.
        Também cria um mapeamento de arquivos para coleções.
        """
        # Prepara diretório para ChromaDB
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Limpa base existente se houver
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            logger.info(f"Diretório ChromaDB já existe, recriando...")
            import shutil
            shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir, exist_ok=True)
        
        # Dicionário para armazenar os vectorstores criados
        vectorstores = {}
        collection_map = {}
        
        # Cria uma coleção para cada arquivo
        for file_path, docs in tqdm(processed_docs.items(), desc="Indexando arquivos"):
            if not docs:
                continue
            
            # Cria um hash para o caminho do arquivo para uso como nome de coleção
            file_hash = self.create_file_hash(file_path)
            collection_name = f"{self.collection_prefix}{file_hash}"
            
            # Adiciona ao mapeamento
            collection_map[file_path] = {
                "collection_name": collection_name,
                "file_hash": file_hash,
                "language": docs[0].metadata.get("language", "unknown"),
                "file_type": docs[0].metadata.get("file_type", "unknown"),
                "num_chunks": len(docs) - 1  # -1 para excluir o documento completo
            }
            
            # Cria vectorstore para este arquivo
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name=collection_name
            )
            
            # Armazena no dicionário
            vectorstores[file_path] = vectorstore
        
        # Salva o mapeamento em um arquivo para referência futura
        self.collection_map = collection_map
        self._save_collection_map()
        
        self.stats["total_collections"] = len(vectorstores)
        logger.info(f"Total de coleções ChromaDB criadas: {self.stats['total_collections']}")
        
        return vectorstores
    
    def _save_collection_map(self) -> None:
        """
        Salva o mapeamento de arquivos para coleções em um arquivo JSON.
        """
        if not self.collection_map:
            return
        
        map_path = os.path.join(self.persist_dir, f"{self.collection_map_name}.json")
        try:
            with open(map_path, 'w') as f:
                json.dump({
                    "files": self.collection_map,
                    "stats": self.stats,
                    "created_at": {
                        "timestamp": time.time(),
                        "human_readable": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }, f, indent=2)
            logger.info(f"Mapeamento de coleções salvo em {map_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar mapeamento de coleções: {e}")


def main(folder_path: str):
    # Inicializa o indexador
    indexer = CodeIndexer()
    
    # Carrega documentos
    docs_by_file = indexer.load_documents_from_folder(folder_path)
    if not docs_by_file:
        logger.error("Nenhum documento encontrado para indexar. Verifique o caminho e os tipos de arquivo.")
        exit(1)
    
    # Processa documentos
    processed_docs = indexer.process_documents(docs_by_file)
    
    # Indexa documentos no ChromaDB
    vectorstores = indexer.index_documents(processed_docs)
    
    # Estatísticas finais
    logger.info("=== Estatísticas de Indexação ===")
    logger.info(f"Arquivos processados: {indexer.stats['loaded']}")
    logger.info(f"Total de arquivos indexados: {indexer.stats['total_files']}")
    logger.info(f"Total de chunks e estruturas extraídas: {indexer.stats['total_chunks']}")
    logger.info(f"Total de coleções ChromaDB: {indexer.stats['total_collections']}")
    logger.info("Indexação concluída com sucesso!")


if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser(
        description="Indexador de código avançado com embeddings locais usando ChromaDB"
    )
    parser.add_argument("--folder", required=True, help="Pasta para indexar")
    args = parser.parse_args()

    logger.info(f"Iniciando indexação de: {args.folder}")
    start_time = time.time()
    main(args.folder)
    elapsed = time.time() - start_time
    logger.info(f"Processo finalizado em {elapsed:.2f} segundos!")