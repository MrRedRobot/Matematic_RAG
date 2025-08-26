import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config.settings import settings
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.rag_engine import RAGEngine
from core.educational_generator import EducationalContentGenerator
from models.llm_manager import LLMManager
from logging_config import setup_logging

logging.basicConfig(level=logging.INFO)
logger = setup_logging()

# Modelos Pydantic para API
class QuestionRequest(BaseModel):
    question: str
    context_filter: Optional[Dict] = None

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict]
    confidence: float
    question: str

class EducationalRequest(BaseModel):
    topic: str
    include_exercises: bool = True
    include_examples: bool = True

class EducationalResponse(BaseModel):
    topic: str
    explanation: str
    examples: Optional[str] = None
    exercises: Optional[str] = None
    sources: List[Dict]
    error: Optional[str] = None

class TopicSummaryResponse(BaseModel):
    topic: str
    summary: str
    confidence: float
    sources_count: int
    related_topics: List[str]

app = FastAPI(
    title="RAG Educational System API",
    description="API para sistema RAG especializado en documentos matemáticos",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_processor: Optional[DocumentProcessor] = None
vector_store: Optional[VectorStore] = None
llm_manager: Optional[LLMManager] = None
rag_engine: Optional[RAGEngine] = None
educational_generator: Optional[EducationalContentGenerator] = None
system_initialized = False

@app.on_event("startup")
async def startup_event():
    """Inicializar el sistema al arrancar"""
    global document_processor, vector_store, llm_manager, system_initialized

    try:
        logger.info("Inicializando sistema RAG...")

        os.makedirs("documents", exist_ok=True)
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)

        document_processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        vector_store = VectorStore()

        llm_manager = LLMManager(model_type="ollama")

        if vector_store.load_vector_store():
            logger.info("Vector store existente cargado")
            await initialize_rag_components()
        else:
            logger.info("No se encontró vector store existente. Esperando carga de documentos.")

        logger.info("Sistema inicializado correctamente")

    except Exception as e:
        logger.error(f"Error inicializando sistema: {str(e)}")

async def initialize_rag_components():
    """Inicializa componentes RAG después de cargar documentos"""
    global rag_engine, educational_generator, system_initialized

    try:
        rag_engine = RAGEngine(llm_manager, vector_store)
        educational_generator = EducationalContentGenerator(rag_engine)
        system_initialized = True
        logger.info("Componentes RAG inicializados")
    except Exception as e:
        logger.error(f"Error inicializando componentes RAG: {str(e)}")
        raise

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "RAG Educational System API",
        "status": "running",
        "system_initialized": system_initialized
    }

@app.get("/health")
async def health_check():
    """Check de salud del sistema"""
    return {
        "status": "healthy",
        "components": {
            "document_processor": document_processor is not None,
            "vector_store": vector_store is not None,
            "llm_manager": llm_manager is not None and llm_manager.is_available(),
            "rag_engine": rag_engine is not None,
            "educational_generator": educational_generator is not None
        }
    }

@app.post("/upload-document")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Subir y procesar documento"""
    if not document_processor:
        raise HTTPException(status_code=500, detail="Sistema no inicializado")

    allowed_extensions = {".pdf", ".txt"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado. Permitidos: {allowed_extensions}"
        )

    try:
        file_path = f"documents/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        background_tasks.add_task(process_document_background, file_path)

        return {
            "message": f"Documento {file.filename} subido exitosamente",
            "filename": file.filename,
            "status": "processing"
        }

    except Exception as e:
        logger.error(f"Error subiendo documento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")

async def process_document_background(file_path: str):
    """Procesa documento en background"""
    global system_initialized

    try:
        logger.info(f"Procesando documento: {file_path}")

        documents = document_processor.load_document(file_path)
        processed_docs = document_processor.process_documents(documents)

        if vector_store.vector_store is None:
            vector_store.create_vector_store(processed_docs)
        else:
            vector_store.vector_store.add_documents(processed_docs)
            if hasattr(vector_store.vector_store, 'persist'):
                vector_store.vector_store.persist()

        if not system_initialized:
            await initialize_rag_components()

        logger.info(f"Documento procesado exitosamente: {file_path}")

    except Exception as e:
        logger.error(f"Error procesando documento en background: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Hacer una pregunta al sistema RAG"""
    if not system_initialized or not rag_engine:
        raise HTTPException(status_code=503, detail="Sistema no inicializado o sin documentos")

    try:
        result = rag_engine.ask_question(request.question)

        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            question=result["question"]
        )

    except Exception as e:
        logger.error(f"Error procesando pregunta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando pregunta: {str(e)}")

@app.post("/generate-educational-content", response_model=EducationalResponse)
async def generate_educational_content(request: EducationalRequest):
    """Generar contenido educativo para un tema"""
    if not system_initialized or not educational_generator:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")

    try:
        content = educational_generator.generate_educational_document(request.topic)
        if "error" in content:
            return EducationalResponse(
                topic=request.topic,
                explanation="",
                error=content["error"]
            )

        response = EducationalResponse(
            topic=content["topic"],
            explanation=content["explanation"],
            sources=content["sources"]
        )

        if request.include_examples:
            response.examples = content["examples"]

        if request.include_exercises:
            response.exercises = content["exercises"]

        return response

    except Exception as e:
        logger.error(f"Error generando contenido educativo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generando contenido: {str(e)}")

@app.get("/topic-summary/{topic}", response_model=TopicSummaryResponse)
async def get_topic_summary(topic: str):
    """Obtener resumen de un tema"""
    if not system_initialized or not educational_generator:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")

    try:
        summary = educational_generator.generate_topic_summary(topic)
        related_topics = educational_generator.suggest_related_topics(topic)

        return TopicSummaryResponse(
            topic=summary["topic"],
            summary=summary["summary"],
            confidence=summary["confidence"],
            sources_count=summary["sources_count"],
            related_topics=related_topics
        )

    except Exception as e:
        logger.error(f"Error generando resumen: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generando resumen: {str(e)}")

@app.get("/list-topics")
async def list_available_topics():
    """Lista temas disponibles basados en documentos cargados"""
    if not system_initialized or not vector_store:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")

    try:
        sample_docs = vector_store.similarity_search("matemáticas", k=20)

        topics = set()
        for doc in sample_docs:
            content_type = doc.metadata.get("content_type", "")
            if content_type:
                topics.update(content_type.split(","))

        clean_topics = [topic.strip() for topic in topics if topic.strip() and topic != "general"]

        return {
            "topics": sorted(clean_topics),
            "total_documents": len(sample_docs)
        }

    except Exception as e:
        logger.error(f"Error listando temas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo temas: {str(e)}")

@app.delete("/reset-system")
async def reset_system():
    """Reiniciar el sistema y limpiar vector store"""
    global system_initialized, rag_engine, educational_generator

    try:
        if vector_store and vector_store.vector_store:
            vector_store.vector_store = None

        rag_engine = None
        educational_generator = None
        system_initialized = False

        import shutil
        if os.path.exists("documents"):
            shutil.rmtree("documents")
        os.makedirs("documents", exist_ok=True)

        if os.path.exists(settings.VECTOR_DB_PATH):
            shutil.rmtree(settings.VECTOR_DB_PATH)
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)

        logger.info("Sistema reiniciado exitosamente")

        return {"message": "Sistema reiniciado exitosamente"}

    except Exception as e:
        logger.error(f"Error reiniciando sistema: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reiniciando sistema: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )