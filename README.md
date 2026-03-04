# CatanLLM 🎲

**LLM plays Catan in real-time — benchmarking small language models for strategic game decisions.**

Built on top of [PyCatan](https://github.com/kjaimez/PyCatan) simulator. The LLM plays turn-by-turn: receives the board state as text → reasons → executes action.

## ¿Por qué esto importa?

Los agentes clásicos de Catan (genéticos, MCTS) entrenan offline durante horas y reproducen estrategias fijas. Este proyecto pregunta algo diferente:

> ¿Puede un LLM pequeño (3B-14B) jugar Catan en tiempo real, turno a turno, razonando solo desde la descripción del estado?

**Resultado inicial (CPU-only, servidor sin GPU):**
- Modelo: `qwen2.5:7b-instruct-q2_K` (3GB)
- **El LLM ganó** la primera partida contra 3 RandomAgents
- 0 fallbacks — JSON válido en cada decisión
- 15.6s/decisión (CPU-only; con RTX 4070 → ~0.3s)

---

## Setup rápido (tu máquina con GPU)

### 1. Clonar repos

```bash
mkdir ~/catan-workspace && cd ~/catan-workspace
git clone https://github.com/kjaimez/PyCatan.git      # lógica del juego
git clone https://github.com/vjrivmon/CatanLLM.git    # este proyecto
cd CatanLLM
```

### 2. Instalar Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &   # arrancar en background
```

Ollama detecta CUDA automáticamente. Con RTX 4070 (12GB VRAM) puedes correr hasta modelos 14B en Q4.

### 3. Bajar modelos

```bash
# Recomendado para empezar (rápido y bueno)
ollama pull qwen2.5:7b-instruct-q4_K_M      # 5.2GB VRAM, ~0.3s/decisión

# Más capaz (cabe en 12GB)
ollama pull qwen2.5:14b-instruct-q4_K_M     # 9.5GB VRAM, ~0.6s/decisión

# Alternativas para comparar
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull mistral:7b-instruct-q4_K_M
ollama pull glm4:9b                          # más cercano a GLM-5 disponible
ollama pull deepseek-r1:8b                   # con razonamiento explícito

# Ver instalados
ollama list
```

### 4. Instalar dependencia Python

```bash
pip install requests
```

### 5. Ajustar rutas (si PyCatan no está en `../PyCatan`)

Edita la variable de ruta en `agents/LLMAgent.py` y `benchmark/runner.py`:
```python
sys.path.insert(0, '/ruta/a/tu/PyCatan')     # línea ~10
sys.path.insert(0, '/ruta/a/tu/CatanLLM')    # línea ~11
```

---

## Uso

### Partida individual

```bash
# LLM (P0) vs 3 RandomAgents
python scripts/run_game.py --model qwen2.5:7b-instruct-q4_K_M

# Dry run — sin LLM, solo prueba la integración
python scripts/run_game.py --dry-run

# Varias partidas
python scripts/run_game.py --model qwen2.5:7b-instruct-q4_K_M --games 5

# Limitar rondas (útil para pruebas rápidas)
python scripts/run_game.py --model qwen2.5:3b --rounds 30
```

### Benchmark multi-modelo

```bash
# Auto-detecta los modelos instalados y los compara
python scripts/multi_model_benchmark.py

# Especificar modelos concretos
python scripts/multi_model_benchmark.py \
  --models qwen2.5:7b-instruct-q4_K_M qwen2.5:14b-instruct-q4_K_M llama3.1:8b \
  --games 3 --rounds 80

# Con baseline aleatorio
python scripts/multi_model_benchmark.py --include-dry-run

# Guardar resultados
python scripts/multi_model_benchmark.py --output results/mi_benchmark.json
```

**Salida del benchmark:**

```
═══════════════════════════════════════════════════════════════════════════
  BENCHMARK MULTI-MODELO — TABLA COMPARATIVA
═══════════════════════════════════════════════════════════════════════════
  Modelo                              Victorias   Avg(s)  Max(s)  Fallback
───────────────────────────────────────────────────────────────────────────
  qwen2.5:14b-instruct-q4_K_M         67% (2/3)   0.61s   1.20s      0.0%
  qwen2.5:7b-instruct-q4_K_M          50% (1/2)   0.31s   0.82s      0.0%
  llama3.1:8b-instruct-q4_K_M         33% (1/3)   0.44s   1.10s      1.2%
  mistral:7b-instruct-q4_K_M          33% (1/3)   0.38s   0.90s      3.4%
═══════════════════════════════════════════════════════════════════════════
  🏆 Mejor modelo: qwen2.5:14b-instruct-q4_K_M
```
*(Valores ilustrativos — los reales los medirás tú con GPU)*

### Tests

```bash
python tests/test_llm_agent.py   # 15 tests, no requiere Ollama
```

---

## Arquitectura

```
CatanLLM/
├── agents/
│   └── LLMAgent.py         # Extiende AgentInterface de PyCatan (10 métodos)
├── llm/
│   ├── client.py           # OllamaClient — REST wrapper localhost:11434
│   ├── prompts.py          # Prompts por fase + parser JSON robusto
│   └── state_encoder.py    # Estado del tablero → texto natural
├── benchmark/
│   ├── runner.py           # BenchmarkRunner — orquesta partidas con métricas
│   └── metrics.py          # GameMetrics, BenchmarkSummary
├── scripts/
│   ├── run_game.py         # CLI: una o varias partidas
│   ├── multi_model_benchmark.py  # Comparar múltiples modelos
│   └── install_model.sh    # Instala Ollama + mejor modelo por RAM disponible
├── docs/
│   └── GPU_SETUP.md        # Guía detallada setup GPU
└── tests/
    └── test_llm_agent.py   # Tests unitarios (dry_run, sin Ollama)
```

---

## Cómo funciona

### Cada turno del LLM

1. **StateEncoder** describe el tablero: recursos, edificios, puntos, acciones posibles
2. **PromptBuilder** genera un prompt específico para la fase del juego
3. **OllamaClient** llama al modelo local (POST /api/generate)
4. La respuesta JSON se parsea y valida
5. Si falla/timeout → fallback aleatorio (tasa registrada)

### Prompts por fase

| Fase | Pregunta al LLM | JSON esperado |
|------|-----------------|---------------|
| `on_build_phase` | ¿Qué construir? | `{"action": "town/city/road/card/none", "node_id": int, "road_to": int}` |
| `on_commerce_phase` | ¿Comerciar? | `{"gives": 0-4, "receives": 0-4}` o `{"action": "skip"}` |
| `on_moving_thief` | ¿Dónde el ladrón? | `{"terrain": 0-18, "player": int}` |
| `on_game_start` | ¿Dónde el primer pueblo? | `{"node_id": int, "road_to": int}` |
| Cards | Monopolio, año abundancia, etc. | JSON específico por carta |

---

## Resultados primera prueba (CPU-only)

| Métrica | Valor |
|---------|-------|
| Hardware | AMD EPYC, 7.6GB RAM, sin GPU |
| Modelo | qwen2.5:7b-instruct-q2_K (3GB) |
| Resultado | **LLM ganó** vs 3 RandomAgents |
| Decisiones LLM | 31 |
| Fallbacks | 0 (0%) |
| Avg decisión | 15.6s (cold: 19s, warm: 8-25s) |
| Tiempo partida | 483s (~8 min, 20 rondas) |

**Con RTX 4070:** esperar ~0.3-0.8s/decisión → partida completa en 2-5 min.

---

## Modelos a probar (prioridad)

| Modelo | VRAM | Calidad esperada | Por qué |
|--------|------|-----------------|---------|
| `qwen2.5:7b-q4_K_M` | 5.2GB | ⭐⭐⭐⭐ | Baseline rápido |
| `qwen2.5:14b-q4_K_M` | 9.5GB | ⭐⭐⭐⭐⭐ | Mejor calidad en 12GB |
| `glm4:9b` | 6.0GB | ⭐⭐⭐⭐ | Más cercano a GLM-5 |
| `deepseek-r1:8b` | 5.2GB | ⭐⭐⭐⭐ | Razonamiento explícito |
| `llama3.1:8b-q4` | 5.0GB | ⭐⭐⭐ | Comparación |
| `mistral:7b-q4` | 4.5GB | ⭐⭐⭐ | Comparación |

> **Nota:** GLM-5 no está en el registry de Ollama todavía. `glm4:9b` es la alternativa más cercana disponible. Se actualizará cuando esté disponible.

---

## Requisitos

- Python 3.10+
- [PyCatan](https://github.com/kjaimez/PyCatan) en `../PyCatan/` (o ajusta la ruta)
- [Ollama](https://ollama.ai) instalado y corriendo
- `pip install requests`

## Licencia

MIT
