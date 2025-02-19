import argparse
import json
import logging
import os
import time

import chardet
import torch
from decouple import config
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import Document
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()


EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_DIMENSION = 768
PERSIST_DIR = "./pinecone_db"
INDEX_NAME = "codebase-index"
ALLOWED_EXTS = {'.py', '.js', '.ts', '.md', '.json', '.yml', '.yaml',
                '.txt', '.sh', '.dockerfile', '.html', '.css', '.vue'}
EXCLUDED_DIRS = {'node_modules', 'venv', '__pycache__', '.git', '.idea', 'env', 'dist', 'build'}
MAX_FILE_SIZE = 50_000_000  # 50 MB
MAX_TOKENS_PER_BATCH = 500000
MAX_METADATA_SIZE = 40000  # 40KB com margem de segurança

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%H:%M:%S'
)

def validate_metadata(metadata: dict) -> dict:
    """Garante que os metadados estejam dentro dos limites do Pinecone."""
    json_str = json.dumps(metadata, separators=(',', ':'))
    while len(json_str.encode('utf-8')) > MAX_METADATA_SIZE:
        if 'source' in metadata:
            base_source = os.path.basename(metadata['source'])

            if base_source == metadata['source'] and len(base_source) > 200:
                metadata['source'] = base_source[:200]
            else:
                metadata['source'] = base_source
            json_str = json.dumps(metadata, separators=(',', ':'))
        else:
            metadata.pop('ext', None)
            json_str = json.dumps(metadata, separators=(',', ':'))
            break
    return metadata

def load_documents_from_folder(folder_path: str):
    """Carrega documentos com filtros otimizados e metadados reduzidos."""
    docs = []
    for root, dirs, files in os.walk(folder_path):

        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in ALLOWED_EXTS:
                continue
            full_path = os.path.join(root, file)
            try:

                if os.path.getsize(full_path) > MAX_FILE_SIZE:
                    logging.debug(f"Arquivo grande ignorado: {full_path}")
                    continue

                with open(full_path, 'rb') as f:
                    raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result.get("encoding") or 'utf-8'


                text = raw_data.decode(encoding, errors="replace").strip()
                if len(text) < 10:
                    continue

                metadata = validate_metadata({
                    "source": os.path.relpath(full_path, folder_path),
                    "ext": ext[1:] if ext else ""
                })

                docs.append(Document(page_content=text, metadata=metadata))

            except Exception as e:
                logging.error(f"Erro ao carregar {full_path}: {str(e)}")
    return docs

def trim_document_for_indexing(doc: Document, snippet_length: int = 300) -> Document:
    """
    Cria um novo documento com metadados reduzidos para indexação no Pinecone.
    Aqui mantemos apenas o 'source' e 'ext' e adicionamos um 'snippet' curto.
    """
    new_metadata = {}
    if "source" in doc.metadata:
        new_metadata["source"] = doc.metadata["source"]
    if "ext" in doc.metadata:
        new_metadata["ext"] = doc.metadata["ext"]

    new_metadata["snippet"] = doc.page_content[:snippet_length]

    new_metadata = validate_metadata(new_metadata)
    return Document(page_content=doc.page_content, metadata=new_metadata)

def main(folder_path: str):
    logging.info("Iniciando carregamento dos documentos...")
    docs = load_documents_from_folder(folder_path)
    logging.info(f"Documentos carregados: {len(docs)}")


    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


    pc = Pinecone(api_key=config("PINECONE_API_KEY"))


    if INDEX_NAME in pc.list_indexes().names():
        pc.delete_index(INDEX_NAME)
        logging.info(f"Índice {INDEX_NAME} removido")
        time.sleep(30)

    logging.info(f"Criando novo índice {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=MODEL_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1")
        )
    )
    time.sleep(60)

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    total_indexed = 0
    batch_size = 100

    try:
        for i in tqdm(range(0, len(docs), batch_size), desc="Indexando lotes"):
            batch = docs[i:i + batch_size]
            chunks = child_splitter.split_documents(batch)

            for j in range(0, len(chunks), 50):
                sub_batch = chunks[j:j + 50]
                sub_batch = [trim_document_for_indexing(doc) for doc in sub_batch]
                try:
                    vectorstore.add_documents(sub_batch)
                    total_indexed += len(sub_batch)
                except Exception as e:
                    logging.error(f"Erro no sub-lote {j}-{j + 50}: {str(e)}")

            logging.debug(f"Lote {i // batch_size} processado: {len(chunks)} chunks")

    except KeyboardInterrupt:
        logging.warning("Processo interrompido pelo usuário!")

    logging.info(f"Total de chunks indexados: {total_indexed}")

    logging.info("Configurando sistema de recuperação...")
    store = LocalFileStore(os.path.join(PERSIST_DIR, "parent_docs"))
    docstore = create_kv_docstore(store)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        byte_store=docstore,
        child_splitter=child_splitter,
        parent_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1600,
            chunk_overlap=300
        ),
        search_kwargs={"k": 5},
        fail_on_missing_parent_document=True
    )

    logging.info("Indexação concluída com sucesso!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexador de código com embeddings locais")
    parser.add_argument("--folder", required=True, help="Pasta para indexar")
    args = parser.parse_args()

    logging.info(f"Iniciando indexação de: {args.folder}")
    main(args.folder)
    logging.info("Processo finalizado!")
