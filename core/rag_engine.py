import logging
from typing import List, Dict, Any
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from .vector_store import VectorStore
from models.llm_manager import LLMManager
from config.settings import settings

logger = logging.getLogger(__name__)

class RAGEngine:
    """Motor principal del sistema RAG optimizado para Ollama y modelos modernos"""

    def __init__(self, llm_manager: LLMManager, vector_store: VectorStore):
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.qa_chain = None
        self._setup_qa_chain()

    def _setup_qa_chain(self):
        """Configura la cadena de pregunta-respuesta optimizada"""
        if not self.llm_manager.is_available():
            raise ValueError("Modelo de lenguaje no disponible")

        if self.llm_manager.model_type == "ollama":
            template = """Eres un asistente experto en matemáticas. Usa el contexto proporcionado para responder la pregunta de forma clara, educativa y completa.

            CONTEXTO:{context}

            PREGUNTA: {question}

            INSTRUCCIONES:
            - Basa tu respuesta únicamente en el contexto proporcionado
            - Si la información no está en el contexto, indica que necesitas más información
            - Explica los conceptos de forma clara y educativa
            - Usa ejemplos cuando sea apropiado
            - Estructura tu respuesta de manera organizada

            RESPUESTA:"""
        else:
            template = """Contexto: {context}
            Pregunta: {question}
            Respuesta clara y educativa:"""

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        try:
            retriever = self.vector_store.get_retriever(
                search_kwargs={"k": settings.MAX_CHUNKS_RETRIEVED}
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm_manager.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": prompt_template,
                },
                return_source_documents=True
            )

            logger.info(f"Cadena QA configurada - Modelo: {self.llm_manager.model_type}, Max chunks: {settings.MAX_CHUNKS_RETRIEVED}")

        except Exception as e:
            logger.error(f"Error configurando cadena QA: {str(e)}")
            raise

    def _prepare_context(self, documents: List[Document], question: str) -> tuple[str, List[Document]]:
        """Prepara el contexto optimizando para el modelo específico"""
        if not documents:
            return "", []

        question_tokens = self.llm_manager.count_tokens(question)

        if self.llm_manager.model_type == "ollama":
            available_tokens = min(settings.MAX_CONTEXT_TOKENS, self.llm_manager.max_model_tokens - 1000)
        else:
            available_tokens = min(300, self.llm_manager.max_model_tokens - 250)

        available_tokens = available_tokens - question_tokens - 150  # Margen para el template

        logger.info(f"Tokens disponibles para contexto: {available_tokens}")

        if available_tokens <= 50:
            logger.warning("Muy pocos tokens disponibles para contexto")
            return "Contexto limitado disponible.", documents[:1]

        context_parts = []
        used_tokens = 0
        used_documents = []

        for i, doc in enumerate(documents):
            doc_tokens = self.llm_manager.count_tokens(doc.page_content)

            if used_tokens + doc_tokens <= available_tokens:
                context_parts.append(f"[Fuente {i+1}]: {doc.page_content}")
                used_tokens += doc_tokens
                used_documents.append(doc)
            else:
                if i == 0 and used_tokens == 0:
                    remaining_tokens = available_tokens - 50
                    if remaining_tokens > 100:
                        ratio = remaining_tokens / doc_tokens
                        target_chars = int(len(doc.page_content) * ratio * 0.8)
                        truncated_content = doc.page_content[:target_chars] + "..."
                        context_parts.append(f"[Fuente {i+1} (parcial)]: {truncated_content}")
                        used_documents.append(Document(
                            page_content=truncated_content,
                            metadata={**doc.metadata, "truncated": True}
                        ))
                        used_tokens += remaining_tokens
                break

        final_context = "\n\n".join(context_parts)
        logger.info(f"Contexto preparado: {len(used_documents)} documentos, ~{used_tokens} tokens")

        return final_context, used_documents

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Procesa una pregunta con manejo optimizado según el modelo"""
        if not self.qa_chain:
            raise ValueError("Cadena QA no configurada")

        try:
            question_tokens = self.llm_manager.count_tokens(question)
            logger.info(f"Procesando pregunta ({self.llm_manager.model_type}): '{question[:100]}...' ({question_tokens} tokens)")

            if self.llm_manager.model_type == "ollama":
                result = self.qa_chain.invoke({"query": question})
                source_info = []
                if "source_documents" in result and result["source_documents"]:
                    for i, doc in enumerate(result["source_documents"]):
                        source_info.append({
                            "id": i + 1,
                            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                            "metadata": doc.metadata,
                            "source": doc.metadata.get("source", f"Documento {i+1}")
                        })

                return {
                    "answer": result["result"],
                    "sources": source_info,
                    "question": question,
                    "confidence": self._calculate_confidence(result),
                    "model_used": self.llm_manager.model_type,
                    "tokens_info": {
                        "question_tokens": question_tokens,
                        "documents_used": len(source_info)
                    }
                }

            else:
                retrieved_docs = self.vector_store.similarity_search(
                    question,
                    k=settings.MAX_CHUNKS_RETRIEVED
                )

                if not retrieved_docs:
                    return {
                        "answer": "No se encontraron documentos relevantes para responder la pregunta.",
                        "sources": [],
                        "question": question,
                        "confidence": 0.0,
                        "model_used": self.llm_manager.model_type
                    }

                context, processed_docs = self._prepare_context(retrieved_docs, question)

                if context.strip():
                    full_prompt = f"""Contexto: {context}

                        Pregunta: {question}

                        Respuesta clara y educativa:"""
                else:
                    full_prompt = f"Pregunta: {question}\n\nRespuesta:"

                response_text = self.llm_manager.generate_response(full_prompt)

                source_info = []
                for i, doc in enumerate(processed_docs):
                    source_info.append({
                        "id": i + 1,
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("source", f"Documento {i+1}"),
                        "truncated": doc.metadata.get("truncated", False)
                    })

                return {
                    "answer": response_text,
                    "sources": source_info,
                    "question": question,
                    "confidence": self._calculate_confidence_manual(response_text, processed_docs),
                    "model_used": self.llm_manager.model_type,
                    "tokens_info": {
                        "question_tokens": question_tokens,
                        "context_length": len(context),
                        "documents_used": len(processed_docs)
                    }
                }

        except Exception as e:
            error_msg = f"Error procesando pregunta: {str(e)}"
            logger.error(error_msg)
            return {
                "answer": "Error al procesar la pregunta. Por favor, intenta con una pregunta más específica o verifica que el sistema esté funcionando correctamente.",
                "sources": [],
                "question": question,
                "confidence": 0.0,
                "model_used": self.llm_manager.model_type,
                "error": str(e)
            }

    def get_relevant_context(self, topic: str, k: int = None) -> List[Document]:
        """Obtiene contexto relevante para un tema específico"""
        if k is None:
            k = settings.MAX_CHUNKS_RETRIEVED

        try:
            documents = self.vector_store.similarity_search(topic, k=k)
            logger.info(f"Contexto obtenido para '{topic}': {len(documents)} documentos")
            return documents
        except Exception as e:
            logger.error(f"Error obteniendo contexto: {str(e)}")
            return []

    def _calculate_confidence(self, result: Dict) -> float:
        """Calcula confianza basada en resultado de la cadena QA"""
        if not result.get("source_documents"):
            return 0.1

        num_sources = len(result["source_documents"])
        answer_length = len(result.get("result", ""))

        base_confidence = min(0.95, (num_sources * 0.25) + (min(answer_length, 1000) / 1000 * 0.5))

        return round(base_confidence, 2)

    def _calculate_confidence_manual(self, answer: str, documents: List[Document]) -> float:
        """Calcula confianza para respuestas generadas manualmente"""
        if not documents or not answer.strip():
            return 0.1

        num_sources = len(documents)
        answer_length = len(answer)

        truncation_penalty = sum(0.1 for doc in documents if doc.metadata.get("truncated", False))

        base_confidence = min(0.9, (num_sources * 0.3) + (min(answer_length, 800) / 800 * 0.4))
        final_confidence = max(0.1, base_confidence - truncation_penalty)

        return round(final_confidence, 2)

    def get_system_status(self) -> Dict[str, Any]:
        """Estado completo del sistema RAG"""
        return {
            "llm_info": self.llm_manager.get_model_info(),
            "vector_store_available": self.vector_store is not None,
            "qa_chain_ready": self.qa_chain is not None,
            "settings": {
                "max_chunks": settings.MAX_CHUNKS_RETRIEVED,
                "chunk_size": settings.CHUNK_SIZE,
                "max_context_tokens": settings.MAX_CONTEXT_TOKENS,
                "max_input_tokens": settings.MAX_INPUT_TOKENS,
                "model_type": settings.llm_model_type
            }
        }