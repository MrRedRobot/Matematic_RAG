import streamlit as st
import requests
import json
from typing import Dict, Any, List
import time
import os
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="RAG Educational System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de la API
API_BASE_URL = "http://localhost:8000"

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }

    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }

    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }

    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }

    .source-box {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Verifica si la API está funcionando"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_status() -> Dict[str, Any]:
    """Obtiene el estado del sistema"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "components": {}}
    except:
        return {"status": "error", "components": {}}

def upload_document(file) -> Dict[str, Any]:
    """Sube un documento a la API"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/upload-document", files=files, timeout=30)

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"Error {response.status_code}: {response.text}"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout - El documento es muy grande o el servidor está ocupado"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def ask_question(question: str, context_filter: Dict = None) -> Dict[str, Any]:
    """Hace una pregunta a la API"""
    try:
        payload = {"question": question}
        if context_filter:
            payload["context_filter"] = context_filter

        response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=30)

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_educational_content(topic: str, include_examples: bool = True, include_exercises: bool = True) -> Dict[str, Any]:
    """Genera contenido educativo"""
    try:
        payload = {
            "topic": topic,
            "include_examples": include_examples,
            "include_exercises": include_exercises
        }

        response = requests.post(f"{API_BASE_URL}/generate-educational-content", json=payload, timeout=60)

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_topic_summary(topic: str) -> Dict[str, Any]:
    """Obtiene resumen de un tema"""
    try:
        response = requests.get(f"{API_BASE_URL}/topic-summary/{topic}", timeout=30)

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_available_topics() -> List[str]:
    """Obtiene lista de temas disponibles"""
    try:
        response = requests.get(f"{API_BASE_URL}/list-topics", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("topics", [])
        return []
    except:
        return []

def main():
    """Función principal de la aplicación Streamlit"""

    st.markdown('<h1 class="main-header">🧮 RAG Educational System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Sistema RAG especializado en documentos matemáticos</p>', unsafe_allow_html=True)

    if not check_api_health():
        st.error("❌ No se puede conectar con la API. Asegúrate de que el servidor esté ejecutándose en http://localhost:8000")
        st.code("python main.py", language="bash")
        st.stop()

    with st.sidebar:
        st.header("📊 Estado del Sistema")

        system_status = get_system_status()

        if system_status["status"] == "error":
            st.error("Sistema no disponible")
        else:
            components = system_status.get("components", {})

            for component, status in components.items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} {component.replace('_', ' ').title()}")

        st.divider()

        available_topics = get_available_topics()
        if available_topics:
            st.subheader("📚 Temas Disponibles")
            for topic in available_topics[:10]:
                if st.button(topic, key=f"topic_{topic}"):
                    st.session_state["selected_topic"] = topic

        st.divider()

        if st.button("🔄 Reiniciar Sistema", type="secondary"):
            with st.spinner("Reiniciando sistema..."):
                try:
                    response = requests.delete(f"{API_BASE_URL}/reset-system", timeout=30)
                    if response.status_code == 200:
                        st.success("Sistema reiniciado exitosamente")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("Error reiniciando sistema")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    tab1, tab2, tab3, tab4 = st.tabs(["📤 Cargar Documentos", "❓ Hacer Preguntas", "📖 Generar Contenido", "📋 Resúmenes"])

    with tab1:
        st.markdown('<div class="section-header">Cargar Documentos Matemáticos</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        📝 <strong>Formatos soportados:</strong> PDF, TXT<br>
        💡 <strong>Tip:</strong> Los documentos se procesarán automáticamente y se dividirán en chunks para facilitar la búsqueda.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Selecciona un documento",
            type=["pdf", "txt"],
            help="Sube documentos matemáticos para que el sistema pueda responder preguntas sobre ellos"
        )

        if uploaded_file is not None:
            st.info(f"Archivo seleccionado: {uploaded_file.name} ({uploaded_file.size} bytes)")

            if st.button("📤 Subir y Procesar Documento", type="primary"):
                with st.spinner("Subiendo y procesando documento..."):
                    result = upload_document(uploaded_file)

                    if result["success"]:
                        st.success("✅ Documento subido exitosamente")
                        st.json(result["data"])

                        st.markdown("""
                        <div class="warning-box">
                        ⏳ <strong>Procesamiento en curso:</strong> El documento se está procesando en segundo plano.
                        Esto puede tomar unos segundos dependiendo del tamaño del archivo.
                        </div>
                        """, unsafe_allow_html=True)

                    else:
                        st.error(f"❌ Error subiendo documento: {result['error']}")

    with tab2:
        st.markdown('<div class="section-header">Hacer Preguntas al Sistema</div>', unsafe_allow_html=True)

        st.markdown("💡 **Ejemplos de preguntas:**")
        example_questions = [
            "¿Qué es una derivada?",
            "Explica el teorema fundamental del cálculo",
            "¿Cómo se calcula la derivada de una función compuesta?",
            "¿Cuáles son las propiedades de los límites?",
            "Explica el concepto de serie convergente"
        ]

        selected_example = st.selectbox("Selecciona un ejemplo o escribe tu propia pregunta:",
                        [""] + example_questions)

        question = st.text_area(
            "Tu pregunta:",
            value=selected_example,
            height=100,
            placeholder="Escribe tu pregunta sobre matemáticas aquí..."
        )

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔍 Preguntar", type="primary", disabled=not question.strip()):
                with st.spinner("Buscando respuesta..."):
                    result = ask_question(question.strip())

                    if result["success"]:
                        data = result["data"]

                        st.markdown('<div class="section-header">Respuesta</div>', unsafe_allow_html=True)
                        st.markdown(data["answer"])

                        confidence = data["confidence"]
                        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                        st.markdown(f"**Confianza:** <span style='color: {confidence_color}'>{confidence:.1%}</span>", unsafe_allow_html=True)

                        if data["sources"]:
                            st.markdown('<div class="section-header">Fuentes</div>', unsafe_allow_html=True)
                            for i, source in enumerate(data["sources"]):
                                with st.expander(f"Fuente {i+1}: {source['source']}"):
                                    st.markdown(f"**Contenido:** {source['content']}")
                                    if source.get("metadata"):
                                        st.json(source["metadata"])

                    else:
                        st.error(f"❌ Error: {result['error']}")

    with tab3:
        st.markdown('<div class="section-header">Generar Contenido Educativo</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            topic_input = st.text_input(
                "Tema para generar contenido:",
                value=st.session_state.get("selected_topic", ""),
                placeholder="Ej: derivadas, integrales, límites..."
            )

        with col2:
            st.write("**Opciones:**")
            include_examples = st.checkbox("Incluir ejemplos", value=True)
            include_exercises = st.checkbox("Incluir ejercicios", value=True)

        if st.button("📖 Generar Contenido Educativo", type="primary", disabled=not topic_input.strip()):
            with st.spinner("Generando contenido educativo..."):
                result = generate_educational_content(
                    topic_input.strip(),
                    include_examples=include_examples,
                    include_exercises=include_exercises
                )

                if result["success"]:
                    data = result["data"]

                    if data.get("error"):
                        st.error(f"❌ {data['error']}")
                    else:
                        st.markdown(f"# 📚 {data['topic'].title()}")

                        if data.get("explanation"):
                            st.markdown("## 📖 Explicación")
                            st.markdown(data["explanation"])

                        if data.get("examples") and include_examples:
                            st.markdown("## 💡 Ejemplos")
                            st.markdown(data["examples"])

                        if data.get("exercises") and include_exercises:
                            st.markdown("## 🏋️ Ejercicios de Práctica")
                            st.markdown(data["exercises"])

                        if data.get("sources"):
                            with st.expander("📋 Ver fuentes utilizadas"):
                                for source in data["sources"]:
                                    st.markdown(f"- **{source['source']}** (Tipo: {source.get('content_type', 'general')})")

                        content_text = f"""# {data['topic'].title()}

                        ## Explicación
                        {data.get('explanation', '')}

                        ## Ejemplos
                        {data.get('examples', '') if include_examples else ''}

                        ## Ejercicios
                        {data.get('exercises', '') if include_exercises else ''}
                        """

                        st.download_button(
                            label="📥 Descargar contenido como MD",
                            data=content_text,
                            file_name=f"{data['topic'].replace(' ', '_')}_contenido_educativo.md",
                            mime="text/markdown"
                        )

                else:
                    st.error(f"❌ Error generando contenido: {result['error']}")

    with tab4:
        st.markdown('<div class="section-header">Resúmenes Rápidos de Temas</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        with col1:
            summary_topic = st.text_input(
                "Tema para resumir:",
                placeholder="Ej: teorema de Pitágoras, ecuaciones diferenciales..."
            )

        with col2:
            if st.button("📋 Generar Resumen", type="primary", disabled=not summary_topic.strip()):
                with st.spinner("Generando resumen..."):
                    result = get_topic_summary(summary_topic.strip())

                    if result["success"]:
                        data = result["data"]

                        st.markdown(f"""
                        <div class="info-box">
                        <h3>📋 {data['topic'].title()}</h3>
                        <p>{data['summary']}</p>
                        <small>
                        <strong>Confianza:</strong> {data['confidence']:.1%} | 
                        <strong>Fuentes:</strong> {data['sources_count']}
                        </small>
                        </div>
                        """, unsafe_allow_html=True)

                        if data.get("related_topics"):
                            st.markdown("### 🔗 Temas Relacionados")
                            cols = st.columns(3)
                            for i, related_topic in enumerate(data["related_topics"][:9]):
                                col_idx = i % 3
                                with cols[col_idx]:
                                    if st.button(f"🔍 {related_topic}", key=f"related_{i}"):
                                        with st.spinner(f"Cargando {related_topic}..."):
                                            related_result = get_topic_summary(related_topic)
                                            if related_result["success"]:
                                                related_data = related_result["data"]
                                                st.markdown(f"""
                                                <div class="success-box">
                                                <h4>{related_data['topic'].title()}</h4>
                                                <p>{related_data['summary'][:200]}...</p>
                                                </div>
                                                """, unsafe_allow_html=True)

                    else:
                        st.error(f"❌ Error: {result['error']}")

        available_topics = get_available_topics()
        if available_topics:
            st.markdown("### 🎯 Resúmenes Rápidos Disponibles")

            topics_to_show = available_topics[:6]
            cols = st.columns(2)

            for i, topic in enumerate(topics_to_show):
                col_idx = i % 2
                with cols[col_idx]:
                    if st.button(f"📋 Resumen: {topic}", key=f"quick_summary_{i}"):
                        with st.spinner(f"Generando resumen de {topic}..."):
                            result = get_topic_summary(topic)
                            if result["success"]:
                                data = result["data"]
                                st.markdown(f"""
                                <div class="source-box">
                                <h4>📋 {data['topic'].title()}</h4>
                                <p>{data['summary']}</p>
                                <small><em>Confianza: {data['confidence']:.1%}</em></small>
                                </div>
                                """, unsafe_allow_html=True)

def show_footer():
    """Muestra información del sistema en el footer"""
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔧 Información del Sistema")
        st.write("- **Framework:** FastAPI + Streamlit")
        st.write("- **LLM:** Modelos gratuitos (HuggingFace)")
        st.write("- **Vector DB:** Chroma")

    with col2:
        st.markdown("### 📚 Funcionalidades")
        st.write("- ✅ Carga de documentos PDF/TXT")
        st.write("- ✅ Búsqueda semántica RAG")
        st.write("- ✅ Generación de contenido educativo")
        st.write("- ✅ Resúmenes automáticos")

    with col3:
        st.markdown("### 🎯 Casos de Uso")
        st.write("- 📖 Estudio de matemáticas")
        st.write("- 👨‍🏫 Creación de material educativo")
        st.write("- 🔍 Consulta rápida de conceptos")
        st.write("- 💡 Generación de ejercicios")

if __name__ == "__main__":
    main()
    show_footer()