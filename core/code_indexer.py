import argparse
import logging
import os
import re
from typing import List, Dict, Optional, Tuple

import chardet
import torch
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

# Configurações
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "code_collection"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

ALLOWED_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".json", ".yml", ".yaml", 
    ".txt", ".sh", ".dockerfile", ".html", ".css", ".vue", ".java", ".c", 
    ".cpp", ".h", ".cs", ".go", ".php", ".rb", ".rust", ".swift"
}

EXCLUDED_DIRS = {
    "node_modules", "venv", ".venv", "__pycache__", ".git", ".idea", 
    "env", "dist", "build", ".pytest_cache", "__snapshots__"
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
    Usa ChromaDB para armazenar embeddings de documentos completos e chunks de texto.
    """
    def __init__(
        self,
        embeddings_model: str = EMBEDDINGS_MODEL,
        persist_dir: str = PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        device: Optional[str] = None
    ):
        self.embeddings_model = embeddings_model
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Determina o device para embeddings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Utilizando device para embeddings: {self.device}")
        
        # Inicializa o modelo de embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Inicializa o text splitter para chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True,
        )
        
        # Contadores para estatísticas
        self.stats = {
            "loaded": 0,
            "skipped_ext": 0,
            "skipped_size": 0,
            "error": 0,
            "total_chunks": 0,
            "total_docs": 0,
        }

    def load_documents_from_folder(self, folder_path: str) -> List[Document]:
        """
        Carrega documentos de código do diretório especificado.
        Retorna uma lista de documentos com metadados aprimorados.
        """
        logger.info(f"Carregando documentos do diretório: {folder_path}")
        docs = []

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

                    docs.append(Document(page_content=text, metadata=metadata))
                    self.stats["loaded"] += 1

                except Exception as e:
                    logger.error(f"Erro ao carregar {full_path}: {str(e)}")
                    self.stats["error"] += 1

        logger.info(f"Documentos carregados: {self.stats['loaded']}")
        logger.info(f"Documentos ignorados por extensão: {self.stats['skipped_ext']}")
        logger.info(f"Arquivos muito grandes ignorados: {self.stats['skipped_size']}")
        logger.info(f"Erros encontrados: {self.stats['error']}")
        
        return docs

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

    def create_search_vectors(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        Cria dois conjuntos de documentos para indexação:
        1. Documentos completos para busca de contexto completo
        2. Chunks para busca semântica mais granular
        
        Adiciona também vetores especiais para estruturas de código importantes.
        """
        full_docs = []  # Documentos completos
        chunked_docs = []  # Chunks de documentos para busca semântica
        
        logger.info("Processando documentos para indexação...")
        
        for doc in tqdm(docs, desc="Criando vetores de busca"):
            ext = doc.metadata.get("ext", "").lower()
            content = doc.page_content
            
            # 1. Adiciona documento completo à lista de documentos completos
            full_docs.append(doc)
            
            # 2. Cria chunks para busca semântica mais granular
            chunks = self.text_splitter.split_text(content)
            source = doc.metadata.get("source", "unknown")
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_id": i,
                    "is_chunk": True,
                    "chunk_size": len(chunk),
                    "entity_type": "text_chunk",
                }
                
                chunked_docs.append(Document(page_content=chunk, metadata=chunk_metadata))
            
            # 3. Para arquivos de código, identifica e adiciona estruturas importantes
            if doc.metadata.get("file_type") == "code":
                self._extract_code_structures(doc, chunked_docs)
            
            # 4. Para arquivos markdown, extrai seções baseadas em cabeçalhos
            elif ext == "md":
                self._extract_markdown_sections(doc, chunked_docs)
        
        self.stats["total_docs"] = len(full_docs)
        self.stats["total_chunks"] = len(chunked_docs)
        
        logger.info(f"Total de documentos completos: {len(full_docs)}")
        logger.info(f"Total de chunks e estruturas extraídas: {len(chunked_docs)}")
        
        return full_docs, chunked_docs

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
            # Extrai classes e funções de Python
            class_pattern = r"class\s+(\w+)[\s\S]*?(?=\nclass|\Z)"
            func_pattern = r"def\s+(\w+)[\s\S]*?(?=\ndef|\nclass|\Z)"
            
            self._extract_pattern_matches(
                doc, content, class_pattern, "class", 100, result_docs
            )
            self._extract_pattern_matches(
                doc, content, func_pattern, "function", 50, result_docs
            )
            
        elif language in ["javascript", "typescript", "react", "react-typescript"]:
            # Extrai padrões de JS/TS (funções, classes, componentes)
            func_patterns = [
                r"function\s+(\w+)[\s\S]*?(?=\n\s*function|\n\s*class|\Z)",
                r"const\s+(\w+)\s*=\s*\([^)]*\)\s*=>[\s\S]*?(?=\n\s*const|\n\s*function|\n\s*class|\Z)",
                r"class\s+(\w+)[\s\S]*?(?=\n\s*class|\Z)",
                r"export\s+(?:default\s+)?(?:const|function|class)\s+(\w+)[\s\S]*?(?=\n\s*export|\Z)",
            ]
            
            for pattern in func_patterns:
                self._extract_pattern_matches(
                    doc, content, pattern, "code_structure", 50, result_docs
                )

    def _extract_pattern_matches(
        self, 
        doc: Document, 
        content: str, 
        pattern: str, 
        entity_type: str, 
        min_length: int, 
        result_docs: List[Document]
    ) -> None:
        """
        Extrai padrões do conteúdo e adiciona à lista de documentos.
        """
        matches = re.finditer(pattern, content)
        for match in matches:
            entity_name = match.group(1)
            entity_content = match.group(0)
            
            if len(entity_content) >= min_length:
                result_docs.append(
                    Document(
                        page_content=entity_content,
                        metadata={
                            **doc.metadata,
                            "entity_type": entity_type,
                            "entity_name": entity_name,
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
                                    "is_chunk": False,
                                    "is_markdown_section": True,
                                },
                            )
                        )
                
                # Inicia nova seção
                current_header = header_match.group(2)
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
                            "is_chunk": False,
                            "is_markdown_section": True,
                        },
                    )
                )

    def index_documents(self, full_docs: List[Document], chunked_docs: List[Document]) -> Chroma:
        """
        Indexa os documentos no ChromaDB.
        Usa duas coleções: uma para documentos completos e outra para chunks.
        """
        # Prepara diretório para ChromaDB
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Limpa base existente se houver
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            logger.info(f"Diretório ChromaDB já existe, recriando...")
            import shutil
            shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir, exist_ok=True)
        
        # Mostra um exemplo para diagnóstico
        if full_docs:
            logger.info(f"Exemplo de documento indexado: {full_docs[0].metadata}")
            logger.info(f"Conteúdo de exemplo (200 caracteres): {full_docs[0].page_content[:200]}")
        
        # Combina os documentos para indexação única
        all_docs = full_docs + chunked_docs
        logger.info(f"Indexando total de {len(all_docs)} documentos...")
        
        # Processa em lotes para evitar problemas de memória
        batch_size = 100
        vectorstore = None
        
        for i in tqdm(range(0, len(all_docs), batch_size), desc="Indexando lotes"):
            batch = all_docs[i : i + batch_size]
            
            if i == 0:
                # Cria o vectorstore na primeira iteração
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_dir,
                    collection_name=self.collection_name
                )
            else:
                # Adiciona documentos nas iterações subsequentes
                vectorstore.add_documents(documents=batch)
            
            # Persiste após cada lote
            if hasattr(vectorstore, "_persist"):
                vectorstore._persist()
        
        logger.info(f"Base ChromaDB criada com sucesso em {self.persist_dir}")
        
        # Verifica a base criada
        try:
            collection = vectorstore.get()
            count = len(collection["ids"])
            logger.info(f"Total de documentos na base ChromaDB: {count}")
        except Exception as e:
            logger.error(f"Erro ao verificar a base ChromaDB: {str(e)}")
        
        return vectorstore


def main(folder_path: str):
    # Inicializa o indexador
    indexer = CodeIndexer()
    
    # Carrega documentos
    docs = indexer.load_documents_from_folder(folder_path)
    if not docs:
        logger.error("Nenhum documento encontrado para indexar. Verifique o caminho e os tipos de arquivo.")
        exit(1)
    
    # Cria vetores de busca
    full_docs, chunked_docs = indexer.create_search_vectors(docs)
    
    # Indexa documentos no ChromaDB
    vectorstore = indexer.index_documents(full_docs, chunked_docs)
    
    # Estatísticas finais
    logger.info("=== Estatísticas de Indexação ===")
    logger.info(f"Documentos processados: {indexer.stats['loaded']}")
    logger.info(f"Documentos completos indexados: {indexer.stats['total_docs']}")
    logger.info(f"Chunks e estruturas extraídas: {indexer.stats['total_chunks']}")
    logger.info(f"Total de vetores criados: {indexer.stats['total_docs'] + indexer.stats['total_chunks']}")
    logger.info("Indexação concluída com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Indexador de código com embeddings locais usando ChromaDB"
    )
    parser.add_argument("--folder", required=True, help="Pasta para indexar")
    args = parser.parse_args()

    logger.info(f"Iniciando indexação de: {args.folder}")
    main(args.folder)
    logger.info("Processo finalizado!")
