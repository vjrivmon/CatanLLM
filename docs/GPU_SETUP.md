# CatanLLM — Setup en tu máquina (RTX 4070 + 64GB RAM)

## 1. Clonar el repo

```bash
# Necesitas PyCatan como dependencia
mkdir -p ~/CatanLLM-workspace
cd ~/CatanLLM-workspace

# Clonar PyCatan (la lógica del juego)
git clone https://github.com/kjaimez/PyCatan.git

# Clonar CatanLLM (este proyecto)
git clone https://github.com/vjrivmon/CatanLLM.git
cd CatanLLM
```

Si los tienes en rutas distintas, edita los `sys.path.insert` en `agents/LLMAgent.py`, `benchmark/runner.py`, etc. para apuntar a tu PyCatan local.

## 2. Instalar Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Ollama detecta automáticamente la GPU NVIDIA y usa CUDA. Con RTX 4070 (12GB VRAM) puedes correr modelos de hasta ~12B parámetros en Q4.

## 3. Arrancar el servidor Ollama

```bash
ollama serve
# Deja esta terminal abierta (o corre en background: ollama serve &)
```

Verificar que funciona:
```bash
curl http://localhost:11434/api/tags
```

## 4. Bajar modelos

### Recomendados por tamaño/calidad (todos caben en 12GB VRAM):

```bash
# Rápido y bueno — ideal para benchmark inicial
ollama pull qwen2.5:7b-instruct-q4_K_M      # 5.2GB VRAM, ~0.3s/decisión

# Equilibrio calidad/velocidad
ollama pull qwen2.5:14b-instruct-q4_K_M     # 9.5GB VRAM, ~0.6s/decisión

# El más capaz que entra en 12GB
ollama pull qwen2.5:14b-instruct-q8_0       # 15GB — necesita offloading parcial

# Alternativas para comparar
ollama pull llama3.1:8b-instruct-q4_K_M     # 5.0GB VRAM
ollama pull mistral:7b-instruct-q4_K_M      # 4.5GB VRAM
ollama pull deepseek-r1:8b                  # 5.2GB VRAM (con razonamiento)

# GLM-4 (lo más cercano a GLM-5 en Ollama por ahora)
ollama pull glm4:9b                         # ~6.0GB VRAM
```

Ver todos los modelos disponibles:
```bash
ollama list
```

## 5. Instalar dependencias Python

```bash
pip install requests
# O con venv:
python3 -m venv .venv && source .venv/bin/activate
pip install requests
```

## 6. Correr una partida

```bash
cd ~/CatanLLM-workspace/CatanLLM

# Partida rápida con qwen2.5:7b (debería ir a ~0.3s/decisión con GPU)
python scripts/run_game.py --model qwen2.5:7b-instruct-q4_K_M

# Ver más detalle (log level DEBUG)
python scripts/run_game.py --model qwen2.5:7b-instruct-q4_K_M --log-level DEBUG

# Dry run para probar la integración sin LLM
python scripts/run_game.py --dry-run
```

## 7. Benchmark multi-modelo

```bash
# Ver benchmark.py para el script multi-modelo
python scripts/multi_model_benchmark.py
```

---

## Ajustar rutas (si PyCatan no está en ../PyCatan)

Edita `agents/LLMAgent.py` línea 10:
```python
sys.path.insert(0, '/ruta/absoluta/a/tu/PyCatan')
```

Y lo mismo en `benchmark/runner.py` línea 13.
