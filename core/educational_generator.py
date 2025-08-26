import logging
from typing import Dict, List, Any
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from .rag_engine import RAGEngine

logger = logging.getLogger(__name__)

class EducationalContentGenerator:
    """Generador de contenido educativo automático"""

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        self.content_templates = self._setup_templates()

    def _setup_templates(self) -> Dict[str, PromptTemplate]:
        """Configura templates para diferentes tipos de contenido"""

        explanation_template = PromptTemplate(
            template="""Basándote en el siguiente contexto matemático, crea una explicación educativa completa del tema.
            Contexto: {context}

            Tema: {topic}

            Crea una explicación que incluya:
            1. Definición clara del concepto
            2. Importancia y aplicaciones
            3. Explicación paso a paso
            4. Conexiones con otros conceptos matemáticos

            Explicación educativa:""",
            input_variables=["context", "topic"]
        )

        example_template = PromptTemplate(
            template="""Usando el siguiente contexto matemático, genera ejemplos prácticos y resueltos.
            Contexto: {context}

            Tema: {topic}

            Genera 2-3 ejemplos que incluyan:
            1. Planteamiento claro del problema
            2. Solución paso a paso
            3. Explicación de cada paso
            4. Verificación del resultado

            Ejemplos:""",
            input_variables=["context", "topic"]
        )

        exercise_template = PromptTemplate(
            template="""Basándote en el contexto matemático proporcionado, crea ejercicios de práctica.
            Contexto: {context}

            Tema: {topic}

            Crea 2-3 ejercicios que incluyan:
            1. Problemas de dificultad progresiva
            2. Instrucciones claras
            3. Diferentes tipos de aplicación del concepto

            Ejercicios de práctica:""",
            input_variables=["context", "topic"]
        )

        return {
            "explanation": explanation_template,
            "examples": example_template,
            "exercises": exercise_template
        }

    def generate_educational_document(self, topic: str) -> Dict[str, Any]:
        """Genera un documento educativo completo sobre un tema"""

        try:
            context_docs = self.rag_engine.get_relevant_context(topic, k=15)

            if not context_docs:
                return {
                    "error": f"No se encontró información sobre el tema: {topic}",
                    "topic": topic
                }

            combined_context = "\n\n".join([doc.page_content for doc in context_docs[:10]])

            educational_content = {
                "topic": topic,
                "explanation": self._generate_explanation(combined_context, topic),
                "examples": self._generate_examples(combined_context, topic),
                "exercises": self._generate_exercises(combined_context, topic),
                "sources": self._extract_source_info(context_docs)
            }

            logger.info(f"Documento educativo generado para: {topic}")
            return educational_content

        except Exception as e:
            logger.error(f"Error generando documento educativo: {str(e)}")
            return {
                "error": f"Error generando contenido: {str(e)}",
                "topic": topic
            }

    def _generate_explanation(self, context: str, topic: str) -> str:
        """Genera explicación educativa"""
        try:
            prompt = self.content_templates["explanation"].format(
                context=context, topic=topic
            )
            explanation = self.rag_engine.llm_manager.generate_response(prompt)
            return explanation
        except Exception as e:
            logger.error(f"Error generando explicación: {str(e)}")
            return f"Error generando explicación para {topic}"

    def _generate_examples(self, context: str, topic: str) -> str:
        """Genera ejemplos prácticos"""
        try:
            prompt = self.content_templates["examples"].format(
                context=context, topic=topic
            )
            examples = self.rag_engine.llm_manager.generate_response(prompt)
            return examples
        except Exception as e:
            logger.error(f"Error generando ejemplos: {str(e)}")
            return f"Error generando ejemplos para {topic}"

    def _generate_exercises(self, context: str, topic: str) -> str:
        """Genera ejercicios de práctica"""
        try:
            prompt = self.content_templates["exercises"].format(
                context=context, topic=topic
            )
            exercises = self.rag_engine.llm_manager.generate_response(prompt)
            return exercises
        except Exception as e:
            logger.error(f"Error generando ejercicios: {str(e)}")
            return f"Error generando ejercicios para {topic}"

    def _extract_source_info(self, documents: List[Document]) -> List[Dict]:
        """Extrae información de las fuentes"""
        sources = []
        seen_sources = set()

        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen_sources:
                sources.append({
                    "source": source,
                    "content_type": doc.metadata.get("content_type", "general"),
                    "chunk_id": doc.metadata.get("chunk_id", "unknown")
                })
                seen_sources.add(source)

        return sources

    def generate_topic_summary(self, topic: str) -> Dict[str, Any]:
        """Genera un resumen rápido de un tema"""
        try:
            summary_question = f"Proporciona un resumen conciso sobre {topic}, incluyendo los conceptos clave y su importancia."

            result = self.rag_engine.ask_question(summary_question)

            return {
                "topic": topic,
                "summary": result["answer"],
                "confidence": result["confidence"],
                "sources_count": len(result["sources"])
            }

        except Exception as e:
            logger.error(f"Error generando resumen: {str(e)}")
            return {
                "topic": topic,
                "summary": f"Error generando resumen: {str(e)}",
                "confidence": 0.0,
                "sources_count": 0
            }

    def suggest_related_topics(self, topic: str) -> List[str]:
        """Sugiere temas relacionados basados en el contexto"""
        try:
            context_docs = self.rag_engine.get_relevant_context(topic, k=10)

            related_concepts = set()

            for doc in context_docs:
                content = doc.page_content.lower()
                content_type = doc.metadata.get("content_type", "")

                import re
                math_patterns = [
                    r'teorema\s+de\s+(\w+)',
                    r'método\s+de\s+(\w+)',
                    r'ecuación\s+(\w+)',
                    r'función\s+(\w+)',
                    r'integral\s+(\w+)',
                    r'derivada\s+(\w+)',
                    r'límite\s+(\w+)',
                    r'serie\s+(\w+)'
                ]

                for pattern in math_patterns:
                    matches = re.findall(pattern, content)
                    related_concepts.update(matches)

            related_topics = list(related_concepts)[:8]

            logger.info(f"Temas relacionados encontrados para '{topic}': {len(related_topics)}")
            return related_topics

        except Exception as e:
            logger.error(f"Error buscando temas relacionados: {str(e)}")
            return []
