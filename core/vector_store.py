from typing import List, Optional, Dict, Any
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """Manejador de base de datos vectorial"""

    def __init__(self, embedding_model: str = None, store_type: str = "chroma"):
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL
        self.store_type = store_type
        self.embeddings = None
        self.vector_store = None
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Inicializa el modelo de embeddings"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'}
            )
            logger.info(f"Modelo de embeddings cargado: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Error cargando embeddings: {str(e)}")
            raise

    def create_vector_store(self, documents: List[Document]) -> None:
        """Crea la base de datos vectorial"""
        if not documents:
            raise ValueError("No se proporcionaron documentos")

        try:
            if self.store_type == "chroma":
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=settings.VECTOR_DB_PATH,
                    collection_name=settings.COLLECTION_NAME
                )
                self.vector_store.persist()

            logger.info(f"Vector store creado exitosamente con {len(documents)} documentos")

        except Exception as e:
            logger.error(f"Error creando vector store: {str(e)}")
            raise

    def load_vector_store(self) -> bool:
        """Carga una base de datos vectorial existente"""
        try:
            if self.store_type == "chroma":
                self.vector_store = Chroma(
                    persist_directory=settings.VECTOR_DB_PATH,
                    embedding_function=self.embeddings,
                    collection_name=settings.COLLECTION_NAME
                )

                if self.vector_store._collection.count() == 0:
                    logger.warning("⚠️ Vector store cargado pero vacío. Ingresa documentos antes de consultar.")
                    return False

            logger.info("Vector store cargado exitosamente")
            return True

        except Exception as e:
            logger.warning(f"No se pudo cargar vector store existente: {str(e)}")
            self.vector_store = None
            return False

    def similarity_search(self, query: str, k: int = 5, filter_dict: Dict = None) -> List[Document]:
        """Búsqueda por similitud"""
        if not self.vector_store:
            raise ValueError("Vector store no inicializado")

        try:
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)

            logger.info(f"Búsqueda completada: {len(results)} resultados encontrados")
            return results

        except Exception as e:
            logger.error(f"Error en búsqueda por similitud: {str(e)}")
            return []

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Búsqueda por similitud con scores"""
        if not self.vector_store:
            raise ValueError("Vector store no inicializado")

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error en búsqueda con score: {str(e)}")
            return []

    def get_retriever(self, search_kwargs: Dict = None):
        """Obtiene un retriever para el RAG"""
        if not self.vector_store:
            raise ValueError("⚠️ Vector store no inicializado o vacío. Ingresa documentos primero.")

        search_kwargs = search_kwargs or {"k": 2}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)