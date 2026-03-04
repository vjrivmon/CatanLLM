# CatanLLM 🎲

**LLM plays Catan in real-time — benchmarking small language models for strategic game decisions.**

Built on top of [PyCatan](https://github.com/kjaimez/PyCatan) simulator. The LLM plays turn-by-turn, receiving the board state as text and deciding each action via an Ollama-served model.

## What this is

Traditional AI for Catan uses genetic algorithms or hand-crafted heuristics — they train offline and replay a pre-computed strategy. This project is different:

**The LLM plays live.** Every single turn:
1. The game state is encoded as natural language
2. The LLM receives it and decides: build a road? Buy a card? Trade?
3. The response (JSON) is parsed and executed
4. We measure how long that takes

**The key question:** Are small models (starting with Qwen 2.5 3B-7B) fast and coherent enough to play a full game of Catan in real-time on consumer hardware?

## Architecture

```
CatanLLM/
├── agents/
│   └── LLMAgent.py         # Extends PyCatan's AgentInterface. Core LLM player.
├── llm/
│   ├── client.py           # Ollama REST client (localhost:11434)
│   ├── prompts.py          # Prompt templates per game phase + JSON parser
│   └── state_encoder.py    # Board state → natural language
├── benchmark/
│   ├── runner.py           # Run games, extract metrics
│   └── metrics.py          # GameMetrics, BenchmarkSummary dataclasses
├── scripts/
│   ├── install_model.sh    # Install Ollama + pull best model for available RAM
│   └── run_game.py         # CLI to run games
└── tests/
    └── test_llm_agent.py   # Unit tests (work without Ollama via dry_run)
```

PyCatan is a dependency at `../PyCatan/` — not modified, used as-is.

## Quick start

### 1. Install Ollama + model
```bash
bash scripts/install_model.sh
```
The script auto-detects available RAM and downloads the best fitting model:
- **>6GB RAM** → `qwen2.5:7b-instruct-q4_K_M` (5.2GB, best quality)
- **4-6GB RAM** → `qwen2.5:7b-instruct-q2_K` (2.9GB, good quality)
- **2.5-4GB RAM** → `qwen2.5:3b-instruct` (1.9GB, fast)
- **<2.5GB RAM** → `qwen2.5:1.5b-instruct` (1GB, minimal)

### 2. Run a game
```bash
# LLM (P0) vs 3 Random agents
python scripts/run_game.py

# Specify model
python scripts/run_game.py --model qwen2.5:7b-instruct-q2_K

# Dry run (no LLM, tests the integration logic)
python scripts/run_game.py --dry-run

# Run 5 games benchmark
python scripts/run_game.py --games 5
```

### 3. Run tests
```bash
cd /root/.openclaw/workspace
python -m pytest CatanLLM/tests/ -v
# Or without pytest:
python CatanLLM/tests/test_llm_agent.py
```

## How it works

### Game phases → LLM prompts

Each PyCatan game phase maps to a specific prompt:

| Phase | Prompt asks | Expected JSON |
|-------|-------------|---------------|
| `on_build_phase` | What to build? | `{"action": "town/city/road/card/none", "node_id": int, "road_to": int}` |
| `on_commerce_phase` | Trade? | `{"gives": 0-4, "receives": 0-4}` or `{"action": "skip"}` |
| `on_moving_thief` | Where to move thief? | `{"terrain": 0-18, "player": -1 or id}` |
| `on_game_start` | Where to place first settlement? | `{"node_id": int, "road_to": int}` |
| `on_monopoly_card_use` | Which resource to monopolize? | `{"material": 0-4}` |
| `on_year_of_plenty_card_use` | Which 2 resources to take? | `{"material": int, "material_2": int}` |

### Fallback system

If the LLM:
- Takes longer than `--timeout` seconds (default: 30s)
- Doesn't respond with parseable JSON
- Raises a connection error

→ Falls back to a simple random/heuristic decision for that turn. The fallback rate is tracked.

### Metrics captured

```
Game 1 Summary:
  Model:          qwen2.5:3b
  LLM Player:     P0
  Winner:         P2 (Random won)
  Total turns:    847
  Total time:     142.3s
  LLM decisions:  211
  Fallbacks:      3 (1.4%)
  Avg LLM time:   0.42s/decision
  Max LLM time:   2.1s
```

## Comparison: LLM vs Genetic Algorithm

| Metric | Genetic Agent | LLM Agent (3B) |
|--------|--------------|----------------|
| Setup time | ~hours of training | model pull once |
| Per-turn latency | <1ms (lookup) | ~0.3-2s |
| Adaptability | fixed strategy | dynamic reasoning |
| Explainability | opaque genes | readable prompts |

The genetic agent's `best_chromosome.json` encodes 93 parameters trained over 40K games. The LLM agent has zero prior Catan knowledge — it reasons from the game state description alone.

## Models tested

| Model | Size | Avg decision | Notes |
|-------|------|--------------|-------|
| `qwen2.5:3b` | 1.9GB | ~0.4s | Fast, occasionally bad moves |
| `qwen2.5:7b-q2_K` | 2.9GB | ~0.8s | Better strategy |
| `qwen2.5:7b-q4_K_M` | 5.2GB | ~1.5s | Best quality |

> GLM-5 8B support coming — currently Ollama registry doesn't include GLM-5. Will add when available.

## Requirements

- Python 3.10+
- [PyCatan](../PyCatan/) at `../PyCatan/`
- [Ollama](https://ollama.ai) (installed via `scripts/install_model.sh`)
- `requests` library: `pip install requests`

## License

MIT
