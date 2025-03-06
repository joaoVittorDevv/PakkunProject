import argparse
import hashlib
import json
import logging
import os
import re
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import time
import chardet
import torch
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME_PREFIX = "file_"
COLLECTION_MAP_NAME = "collection_map"
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300

ALLOWED_EXTS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".md",
    ".json",
    ".yml",
    ".yaml",
    ".txt",
    ".sh",
    ".dockerfile",
    ".html",
    ".css",
    ".vue",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".cs",
    ".go",
    ".php",
    ".rb",
    ".rust",
    ".swift",
}

EXCLUDED_DIRS = {
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    ".git",
    ".idea",
    "env",
    "dist",
    "build",
    ".pytest_cache",
    "__snapshots__",
    "chroma_db",
}

MAX_FILE_SIZE = 100_000_000  # 100 MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class CodeIndexer:
    """
    Classe responsável por indexar código fonte e documentação.
    Usa ChromaDB para armazenar embeddings em uma coleção unificada, usando metadados para diferenciar os arquivos.
    """

    def __init__(
        self,
        embeddings_model: str = EMBEDDINGS_MODEL,
        persist_dir: str = PERSIST_DIR,
        collection_prefix: str = COLLECTION_NAME_PREFIX,
        device: Optional[str] = None,
    ):
        self.embeddings_model = embeddings_model
        self.persist_dir = persist_dir
        self.collection_prefix = collection_prefix
        self.collection_map_name = COLLECTION_MAP_NAME

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Utilizando device para embeddings: {self.device}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

        self._init_text_splitters()

        self.stats = {
            "loaded": 0,
            "skipped_ext": 0,
            "skipped_size": 0,
            "error": 0,
            "total_chunks": 0,
            "total_files": 0,
            "total_collections": 0,
        }

        self.collection_map = {}

    def _init_text_splitters(self):
        """
        Inicializa text splitters específicos para diferentes linguagens.
        """
        self.language_splitters = {
            "python": RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""],
            ),
            "javascript": RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""],
            ),
            "typescript": RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""],
            ),
            "react": RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""],
            ),
            "react-typescript": RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""],
            ),
        }

        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    def load_documents_from_folder(self, folder_path: str) -> Dict[str, List[Document]]:
        """
        Carrega documentos de código do diretório especificado.
        Retorna um dicionário mapeando caminho de arquivo para lista de Document(s).
        """
        logger.info(f"Carregando documentos do diretório: {folder_path}")
        docs_by_file: Dict[str, List[Document]] = {}

        file_paths: List[str] = []
        for root, dirs, files in os.walk(folder_path):

            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in ALLOWED_EXTS:
                    self.stats["skipped_ext"] += 1
                    continue
                full_path = os.path.join(root, file)
                file_paths.append(full_path)

        def process_file(full_path: str) -> Tuple[str, Optional[Document]]:
            rel_path = os.path.relpath(full_path, folder_path)
            try:

                if os.path.getsize(full_path) > MAX_FILE_SIZE:
                    logger.warning(f"Arquivo grande ignorado: {full_path}")
                    return ("skipped_size", None)

                with open(full_path, "rb") as f:
                    raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result.get("encoding") or "utf-8"
                text = raw_data.decode(encoding, errors="replace").strip()
                if len(text) < 10:
                    return ("skipped_ext", None)
                ext = os.path.splitext(full_path)[1].lower()
                file_type, language = self._get_file_metadata(ext)
                metadata = {
                    "source": rel_path,
                    "ext": ext[1:] if ext else "",
                    "full_path": full_path,
                    "file_size": len(text),
                    "file_type": file_type,
                    "language": language,
                    "is_chunk": False,
                }
                document = Document(page_content=text, metadata=metadata)
                return ("loaded", (rel_path, document))
            except Exception as e:
                logger.error(f"Erro ao carregar {full_path}: {e}")
                return ("error", None)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, path) for path in file_paths]
            for future in as_completed(futures):
                status, result = future.result()
                if status == "loaded" and result:
                    rel_path, document = result
                    docs_by_file[rel_path] = [document]
                    self.stats["loaded"] += 1
                elif status == "skipped_ext":
                    self.stats["skipped_ext"] += 1
                elif status == "skipped_size":
                    self.stats["skipped_size"] += 1
                elif status == "error":
                    self.stats["error"] += 1

        logger.info(f"Documentos carregados: {self.stats['loaded']}")
        logger.info(f"Documentos ignorados por extensão: {self.stats['skipped_ext']}")
        logger.info(f"Arquivos muito grandes ignorados: {self.stats['skipped_size']}")
        logger.info(f"Erros encontrados durante a leitura: {self.stats['error']}")

        return docs_by_file

    def _get_file_metadata(self, ext: str) -> Tuple[str, str]:
        """
        Determina o tipo e a linguagem de um arquivo com base na sua extensão.
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

            return ("code", "unknown")

    def index_documents(
        self, processed_docs: Dict[str, List[Document]]
    ) -> Dict[str, Chroma]:
        """
        Indexa os documentos no ChromaDB, criando uma única coleção unificada para todos os arquivos.
        Retorna um dicionário de coleções (aqui vazio, pois é usada coleção única persistida em disco).
        """
        os.makedirs(self.persist_dir, exist_ok=True)
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            logger.info(f"Diretório ChromaDB já existe, recriando base do zero...")
            import shutil

            shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir, exist_ok=True)

        collection_map: Dict[str, Dict[str, Any]] = {}
        all_docs: List[Document] = []

        for file_path, docs in processed_docs.items():
            if not docs:
                continue
            all_docs.extend(docs)
            file_hash = self.create_file_hash(file_path)
            collection_map[file_path] = {
                "collection_name": "code_collection",
                "file_hash": file_hash,
                "language": docs[0].metadata.get("language", "unknown"),
                "file_type": docs[0].metadata.get("file_type", "unknown"),
                "num_chunks": len(docs) - 1,
            }

        if all_docs:
            Chroma.from_documents(
                documents=all_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name="code_collection",
            )

        self.collection_map = collection_map
        self.stats["total_files"] = len(collection_map)
        self.stats["total_collections"] = 1 if all_docs else 0
        self._save_collection_map()

        logger.info(
            f"Total de coleções ChromaDB criadas: {self.stats['total_collections']}"
        )
        return {}

    def _save_collection_map(self) -> None:
        """
        Salva o mapeamento de arquivos para coleções em um arquivo JSON.
        """
        if not self.collection_map:
            return
        map_path = os.path.join(self.persist_dir, f"{self.collection_map_name}.json")
        try:
            with open(map_path, "w") as f:
                json.dump(
                    {
                        "files": self.collection_map,
                        "stats": self.stats,
                        "created_at": {
                            "timestamp": time.time(),
                            "human_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Mapeamento de coleções salvo em {map_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar mapeamento de coleções: {e}")

    def create_file_hash(self, file_path: str) -> str:
        """
        Gera um hash único para o caminho de um arquivo (utilizado para nomes de coleções ou identificação de arquivos).
        """
        return hashlib.md5(file_path.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Indexador de código usando ChromaDB.")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Pasta contendo os arquivos a serem indexados.",
    )

    args = parser.parse_args()

    try:
        indexer = CodeIndexer()
        logger.info(f"Iniciando indexação na pasta: {args.folder}")

        docs_by_file = indexer.load_documents_from_folder(args.folder)

        processed_docs = docs_by_file

        indexer.index_documents(processed_docs)

        logger.info("Indexação concluída com sucesso!")
    except Exception as e:
        logger.error(f"Erro durante a execução do indexador: {e}", exc_info=True)
