# core/document_processor.py
import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Procesador de documentos matemáticos"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
            length_function=len
        )

    def load_document(self, file_path: str) -> List[Document]:
        """Carga un documento desde archivo"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        if path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        elif path.suffix.lower() in ['.txt']:
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Formato de archivo no soportado: {path.suffix}")

        try:
            documents = loader.load()
            logger.info(f"Documento cargado exitosamente: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error cargando documento {file_path}: {str(e)}")
            raise

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Procesa y divide documentos en chunks"""
        processed_docs = []

        for doc in documents:
            cleaned_content = self._clean_text(doc.page_content)
            doc.page_content = cleaned_content

            chunks = self.text_splitter.split_documents([doc])

            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': f"{doc.metadata.get('source', 'unknown')}_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'content_type': self._detect_content_type(chunk.page_content)
                })
                processed_docs.append(chunk)

        logger.info(f"Documentos procesados: {len(processed_docs)} chunks creados")
        return processed_docs

    def _clean_text(self, text: str) -> str:
        """Limpia y preprocesa el texto"""
        import re

        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'([.!?])\s*\n', r'\1\n\n', text)

        return text.strip()

    def _detect_content_type(self, text: str) -> str:
        """Detecta el tipo de contenido matemático"""
        import re

        patterns = {
            'theorem': r'(teorema|theorem|lema|lemma|proposición|proposition)',
            'proof': r'(demostración|proof|prueba)',
            'definition': r'(definición|definition|definimos|define)',
            'example': r'(ejemplo|example|por ejemplo|for example)',
            'formula': r'[\$\\\(\)\[\]{}∑∫∂∇±×÷≤≥≠≈∞]',
            'exercise': r'(ejercicio|exercise|problema|problem)'
        }

        text_lower = text.lower()
        content_types = []

        for content_type, pattern in patterns.items():
            if re.search(pattern, text_lower):
                content_types.append(content_type)

        return ','.join(content_types) if content_types else 'general'
