"""
OllamaClient - Wrapper para el servidor Ollama local.
Llama a la API REST en http://localhost:11434.
"""
import json
import time
import requests


class OllamaClient:
    def __init__(self, model: str = 'qwen2.5:3b', base_url: str = 'http://localhost:11434'):
        self.model = model
        self.base_url = base_url
        self.total_tokens = 0
        self.total_calls = 0
        self.total_time = 0.0

    def generate(self, prompt: str, timeout: int = 30, temperature: float = 0.3) -> str:
        """
        Genera una respuesta del LLM dado un prompt.
        Retorna el texto generado o lanza excepción si falla.
        """
        start = time.time()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 256,
            }
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - start
            self.total_time += elapsed
            self.total_calls += 1
            tokens = data.get('eval_count', 0) + data.get('prompt_eval_count', 0)
            self.total_tokens += tokens
            return data.get('response', '').strip()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"LLM no respondió en {timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("No se puede conectar con Ollama en localhost:11434. ¿Está arrancado? Ejecuta: ollama serve")

    def chat(self, messages: list, timeout: int = 30, temperature: float = 0.3) -> str:
        """
        Interfaz de chat (lista de mensajes role/content).
        """
        start = time.time()
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 256,
            }
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - start
            self.total_time += elapsed
            self.total_calls += 1
            return data.get('message', {}).get('content', '').strip()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"LLM no respondió en {timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("No se puede conectar con Ollama en localhost:11434. Ejecuta: ollama serve")

    def is_available(self) -> bool:
        """Comprueba si Ollama está corriendo."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list:
        """Lista los modelos disponibles en Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m['name'] for m in resp.json().get('models', [])]
        except Exception:
            return []

    def stats(self) -> dict:
        """Retorna estadísticas acumuladas de uso."""
        return {
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens,
            'total_time_s': round(self.total_time, 2),
            'avg_time_per_call_s': round(self.total_time / max(self.total_calls, 1), 2),
            'model': self.model,
        }
