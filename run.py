import os
import sys
import subprocess
import time
import signal
import argparse
import importlib.util
from pathlib import Path

def check_requirements():
    """Verifica que las dependencias estén instaladas"""

    required_packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'streamlit': 'streamlit',
        'langchain-community': 'langchain_community',
        'langchain-core': 'langchain_core',
        'transformers': 'transformers',
        'sentence-transformers': 'sentence_transformers',
        'chromadb': 'chromadb'
    }

    missing_packages = []

    for pip_name, import_name in required_packages.items():
            if importlib.util.find_spec(import_name) is None:
                missing_packages.append(pip_name)

    if missing_packages:
        print("❌ Faltan las siguientes dependencias:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstala las dependencias con:")
        print("pip install -r requirements.txt")
        return False

    return True

def start_api_server(port=8000):
    """Inicia el servidor FastAPI"""

    print(f"🚀 Iniciando API FastAPI en puerto {port}...")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ]

    return subprocess.Popen(cmd)

def start_streamlit_app(port=8501):
    """Inicia la aplicación Streamlit"""

    print(f"🎨 Iniciando interfaz Streamlit en puerto {port}...")

    cmd = [
        sys.executable, "-m", "streamlit",
        "run", "streamlit_app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0"
    ]

    return subprocess.Popen(cmd)

def wait_for_api(host="localhost", port=8000, timeout=10800):
    """Espera a que la API esté disponible"""

    import requests

    url = f"http://{host}:{port}/health"

    for i in range(timeout):
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                print("✅ API disponible")
                return True
        except:
            pass

        time.sleep(1)
        if i % 5 == 0:
            print(f"⏳ Esperando API... ({i}/{timeout})")

    return False

def main():
    parser = argparse.ArgumentParser(description="RAG Educational System")
    parser.add_argument("--api-port", type=int, default=8000, help="Puerto para API FastAPI")
    parser.add_argument("--ui-port", type=int, default=8501, help="Puerto para interfaz Streamlit")
    parser.add_argument("--api-only", action="store_true", help="Solo ejecutar API")
    parser.add_argument("--ui-only", action="store_true", help="Solo ejecutar interfaz")
    parser.add_argument("--check-deps", action="store_true", help="Solo verificar dependencias")

    args = parser.parse_args()

    if args.check_deps:
        if check_requirements():
            print("✅ Todas las dependencias están instaladas")
            return 0
        else:
            return 1

    if not check_requirements():
        return 1

    os.makedirs("documents", exist_ok=True)
    os.makedirs("vector_store", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    processes = []

    try:
        if not args.ui_only:
            api_process = start_api_server(args.api_port)
            processes.append(api_process)

            if not args.api_only:
                if not wait_for_api(port=args.api_port):
                    print("❌ No se pudo conectar con la API")
                    return 1

        if not args.api_only:
            streamlit_process = start_streamlit_app(args.ui_port)
            processes.append(streamlit_process)

        print("\n" + "="*50)
        print("🎉 Sistema RAG Educativo iniciado exitosamente")
        print("="*50)

        if not args.ui_only:
            print(f"📡 API FastAPI: http://localhost:{args.api_port}")
            print(f"📚 Documentación: http://localhost:{args.api_port}/docs")

        if not args.api_only:
            print(f"🎨 Interfaz Streamlit: http://localhost:{args.ui_port}")

        print("\nPresiona Ctrl+C para detener el sistema")
        print("="*50)

        while True:
            time.sleep(1)

            for process in processes[:]:
                if process.poll() is not None:
                    print(f"⚠️  Proceso terminado inesperadamente (código: {process.returncode})")
                    processes.remove(process)

            if not processes:
                print("❌ Todos los procesos han terminado")
                break

    except KeyboardInterrupt:
        print("\n🛑 Deteniendo sistema...")

    finally:
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        print("✅ Sistema detenido")

if __name__ == "__main__":
    exit(main())