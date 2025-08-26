import os
import sys
import subprocess
import platform
from pathlib import Path

BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / "venv"

def run_command(cmd, description="", use_venv=False):
    """Ejecuta un comando y maneja errores"""
    python_exec = str(VENV_DIR / "bin" / "python") if use_venv and platform.system() != "Windows" else str(VENV_DIR / "Scripts" / "python.exe")
    if cmd.startswith("python "):
        cmd = cmd.replace("python", python_exec, 1)

    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, check=True, shell=True,
                                capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Salida: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Verifica la versión de Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Se requiere Python 3.8 o superior")
        print(f"Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detectado")
    return True

def create_or_verify_venv():
    """Crea el entorno virtual si no existe"""
    if not VENV_DIR.exists():
        print("📦 Creando entorno virtual en ./venv ...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print("✅ Entorno virtual creado")
    else:
        print("ℹ️  Entorno virtual ya existe")

    # Avisar si no se está dentro del venv
    if sys.prefix != str(VENV_DIR):
        print("\n⚠️  No estás dentro del entorno virtual.")
        print("   Para activarlo ejecuta:")
        if platform.system() == "Windows":
            print(r"   venv\Scripts\activate")
        else:
            print("   source venv/bin/activate")
        print("   Luego vuelve a ejecutar este instalador.\n")
        return False
    return True

def install_requirements():
    """Instala las dependencias del requirements.txt"""

    requirements_content = """
    fastapi==0.115.5
    uvicorn==0.24.0
    streamlit==1.28.1
    langchain==0.3.9
    langchain-community==0.3.9
    langchain-core==0.3.21
    langchain-chroma==0.2.2
    langchain-openai==0.2.10
    langchain-text-splitters==0.3.2
    langchain-huggingface==0.1.0
    langsmith==0.1.147
    pypdf==4.1.0
    python-multipart==0.0.6
    unstructured==0.10.30
    chromadb==0.4.18
    sentence-transformers==2.6.0
    torch==2.2.0
    transformers==4.45.0
    python-dotenv==1.0.0
    pydantic==2.7.4
    pydantic-settings==2.10.1
    requests==2.31.0
    psutil==5.9.6
    """

    with open("requirements.txt", "w") as f:
        f.write(requirements_content.strip())

    print("📦 Instalando dependencias...")

    if not run_command(
        "python -m pip install --upgrade pip setuptools==70.0.0 setuptools-scm wheel",
        "Actualizando pip/setuptools/wheel",
        use_venv=True
    ):
        return False

    if not run_command(
        "python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu",
        "Instalando dependencias de requirements.txt",
        use_venv=True
    ):
        return False

    return True

def create_project_structure():
    """Crea la estructura de directorios del proyecto"""
    directories = [
        "core", "models", "utils",
        "documents", "vector_store",
        "logs",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        if directory in ["core", "models", "utils"]:
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write(f'"""Módulo {directory}"""')
    print("✅ Estructura de proyecto creada")

def create_env_file():
    """Crea archivo .env con configuración por defecto compatible con Settings"""

    env_content = """# Configuración del Sistema RAG Educativo
    # API
    API_HOST=0.0.0.0
    API_PORT=8000

    # Streamlit
    STREAMLIT_PORT=8501

    # LLM
    LLM_MODEL_TYPE=huggingface
    LLM_TEMPERATURE=0.7
    LLM_MAX_TOKENS=1000

    # Vector store
    VECTOR_DB_PATH=vector_store
    COLLECTION_NAME=math_documents
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

    # Procesamiento de documentos
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200

    # Logging
    LOG_LEVEL=INFO
    LOG_FILE=logs/rag_system.log

    #HuggingFace
    HF_TOKEN=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

    #chroma
    CHROMA_TELEMETRY_ENABLED=False
    """

    env_path = ".env"

    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write(env_content)
        print("✅ Archivo .env creado")
    else:
        print("ℹ️  Archivo .env ya existe, se mantiene sin cambios")


def main():
    print("🚀 Instalador del Sistema RAG Educativo")
    print("="*50)

    create_project_structure()
    create_env_file()

    if not check_python_version():
        return 1

    if not create_or_verify_venv():
        return 1

    if not install_requirements():
        print("❌ Error instalando dependencias")
        return 1

    print("\n" + "="*50)
    print("✅ Instalación completada exitosamente")
    print("="*50)
    print("\nPara iniciar el sistema ejecuta:")
    print("python run.py")
    print("\nPara verificar la instalación:")
    print("python run.py --check-deps")
    return 0

if __name__ == "__main__":
    exit(main())
