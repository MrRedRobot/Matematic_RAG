import logging
from typing import Optional, Dict, Any
from langchain_huggingface import HuggingFacePipeline
from langchain_community.llms import Ollama
from transformers import pipeline, AutoTokenizer
import torch
import requests
from config.settings import settings

logger = logging.getLogger(__name__)

class LLMManager:
    """Manejador de modelos optimizado para Ollama y HuggingFace"""

    def __init__(self, model_type: str = "ollama"):
        self.model_type = model_type
        self.llm = None
        self.tokenizer = None
        self.max_model_tokens = 8192
        self.model_config = None
        self._initialize_model()

    def _initialize_model(self):
        """Inicializa el modelo de lenguaje"""
        try:
            if self.model_type == "ollama":
                self._setup_ollama_model()
            elif self.model_type == "huggingface":
                self._setup_huggingface_model()
            else:
                raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")

        except Exception as e:
            logger.error(f"Error inicializando modelo {self.model_type}: {str(e)}")
            if self.model_type == "ollama":
                logger.info("Intentando fallback a HuggingFace...")
                self.model_type = "huggingface"
                self._setup_huggingface_model()

    def _check_ollama_availability(self) -> bool:
        """Verifica si Ollama está disponible"""
        try:
            response = requests.get(f"{settings.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _setup_ollama_model(self):
        """Configura modelo Ollama con verificaciones"""
        if not self._check_ollama_availability():
            raise Exception("Ollama no está disponible. Asegúrate de que esté ejecutándose.")

        model_priorities = ["ollama_llama3", "ollama_qwen", "ollama_phi3"]

        for model_key in model_priorities:
            try:
                model_config = settings.LLM_MODELS[model_key]
                model_name = model_config["model_name"]

                logger.info(f"Probando modelo Ollama: {model_name}")

                if self._check_ollama_model_exists(model_name):
                    self.llm = Ollama(
                        model=model_name,
                        base_url=settings.ollama_host,
                        temperature=model_config["temperature"],
                        timeout=settings.ollama_timeout
                    )

                    self.model_config = model_config
                    self.max_model_tokens = model_config["context_window"]

                    test_response = self.llm("Test")
                    logger.info(f"Modelo Ollama cargado exitosamente: {model_name}")
                    logger.info(f"Context window: {self.max_model_tokens} tokens")
                    return

            except Exception as e:
                logger.warning(f"Error con modelo {model_key}: {e}")
                continue
        raise Exception("No se pudo cargar ningún modelo de Ollama")

    def _check_ollama_model_exists(self, model_name: str) -> bool:
        """Verifica si un modelo específico existe en Ollama"""
        try:
            response = requests.get(f"{settings.ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                available_models = [model["name"] for model in models.get("models", [])]
                return model_name in available_models
            return False
        except Exception as e:
            logger.warning(f"No se pudo verificar modelo {model_name}: {e}")
            return False

    def _setup_huggingface_model(self):
        """Configura modelo HuggingFace como fallback"""
        logger.info("Configurando HuggingFace como fallback...")

        models_to_try = [
            ("microsoft/DialoGPT-medium", 1024),
            ("distilgpt2", 1024),
            ("gpt2", 1024)
        ]

        for model_name, context_size in models_to_try:
            try:
                logger.info(f"Probando modelo HuggingFace: {model_name}")

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token

                text_generator = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=tokenizer,
                    device=-1,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    truncation=True,
                )

                self.tokenizer = tokenizer
                self.llm = HuggingFacePipeline(
                    pipeline=text_generator,
                    model_kwargs={"return_full_text": False}
                )
                self.max_model_tokens = context_size

                logger.info(f"Modelo HuggingFace cargado: {model_name}")
                return

            except Exception as e:
                logger.warning(f"Error con {model_name}: {e}")
                continue

        raise Exception("No se pudo cargar ningún modelo")

    def count_tokens(self, text: str) -> int:
        """Cuenta tokens de forma precisa"""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
            except Exception:
                pass

        if self.model_type == "ollama":
            return max(1, len(text) // 4)
        else:
            return max(1, len(text) // 4)

    def validate_input_size(self, prompt: str) -> tuple[bool, str]:
        """Valida tamaño del input con límites apropiados"""
        token_count = self.count_tokens(prompt)

        if self.model_type == "ollama":
            max_output_tokens = self.model_config.get("max_tokens", 2000) if self.model_config else 2000
            available_tokens = self.max_model_tokens - max_output_tokens - 100  # Margen
        else:
            max_output_tokens = 200
            available_tokens = self.max_model_tokens - max_output_tokens - 50

        if token_count <= available_tokens:
            return True, f"OK: {token_count}/{available_tokens} tokens"
        else:
            return False, f"Excede límite: {token_count} tokens > {available_tokens} disponibles"

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Genera respuesta optimizada según el modelo"""
        if not self.llm:
            raise ValueError("Modelo no inicializado")

        try:
            is_valid, message = self.validate_input_size(prompt)
            logger.info(f"Validación input: {message}")

            if self.model_type == "ollama":
                response = self.llm(prompt, **kwargs)
            else:
                if not is_valid:
                    available_tokens = 400
                    safe_prompt = self._truncate_prompt(prompt, available_tokens)
                    logger.warning("Prompt truncado para HuggingFace")
                else:
                    safe_prompt = prompt

                response = self.llm(safe_prompt, **kwargs)

            logger.info(f"Respuesta generada exitosamente ({self.model_type})")
            return response

        except Exception as e:
            error_msg = f"Error generando respuesta: {str(e)}"
            logger.error(error_msg)
            return f"Error al procesar la consulta con {self.model_type}. Intenta con una pregunta más específica."

    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """Trunca prompt preservando coherencia"""
        current_tokens = self.count_tokens(prompt)

        if current_tokens <= max_tokens:
            return prompt

        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(prompt, truncation=True, max_length=max_tokens)
                return self.tokenizer.decode(tokens, skip_special_tokens=True)
            except Exception:
                pass

        ratio = max_tokens / current_tokens
        target_chars = int(len(prompt) * ratio * 0.8)
        return prompt[:target_chars] + "..."

    def is_available(self) -> bool:
        """Verifica disponibilidad del modelo"""
        return self.llm is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Información detallada del modelo"""
        info = {
            "model_type": self.model_type,
            "max_tokens": self.max_model_tokens,
            "available": self.is_available(),
            "has_tokenizer": self.tokenizer is not None
        }

        if self.model_config:
            info["model_name"] = self.model_config.get("model_name", "Unknown")
            info["temperature"] = self.model_config.get("temperature", 0.1)
            info["max_output_tokens"] = self.model_config.get("max_tokens", "Unknown")

        return info
