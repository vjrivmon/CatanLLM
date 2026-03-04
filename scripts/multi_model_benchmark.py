"""
multi_model_benchmark.py — Compara varios modelos LLM jugando Catan.

Corre N partidas con cada modelo y genera una tabla comparativa de:
- Velocidad (tiempo medio por decisión)
- Calidad (victorias vs RandomAgent)
- Fiabilidad (tasa de fallbacks)

Uso:
    python scripts/multi_model_benchmark.py
    python scripts/multi_model_benchmark.py --games 3 --rounds 50
    python scripts/multi_model_benchmark.py --models qwen2.5:3b qwen2.5:7b llama3.1:8b
    python scripts/multi_model_benchmark.py --output results/benchmark_2026.json
"""
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, '/root/.openclaw/workspace/PyCatan')
sys.path.insert(0, '/root/.openclaw/workspace/CatanLLM')

from benchmark.runner import BenchmarkRunner
from benchmark.metrics import BenchmarkSummary
from llm.client import OllamaClient


# ── Modelos por defecto (edita esta lista según tu hardware) ────────────────

# Para RTX 4070 (12GB VRAM) — todos caben:
MODELS_GPU = [
    'qwen2.5:7b-instruct-q4_K_M',   # Rápido y bueno (~0.3s/dec)
    'qwen2.5:14b-instruct-q4_K_M',  # Más capaz (~0.6s/dec)
    'llama3.1:8b-instruct-q4_K_M',  # Comparación
    'mistral:7b-instruct-q4_K_M',   # Comparación
    'glm4:9b',                        # El más cercano a GLM-5
]

# Para servidor CPU-only (4.6GB RAM) — solo modelos pequeños:
MODELS_CPU = [
    'qwen2.5:7b-instruct-q2_K',     # 3.0GB — ya instalado
    'qwen2.5:3b-instruct',           # 1.9GB — más rápido
]


def detect_available_models(client: OllamaClient, wanted: list[str]) -> list[str]:
    """Filtra la lista a solo los modelos realmente instalados."""
    installed = client.list_models()
    available = []
    for model in wanted:
        if any(model in m or m in model for m in installed):
            available.append(model)
        else:
            print(f"  ⚠️  Modelo no instalado (skip): {model}")
            print(f"       → Para instalarlo: ollama pull {model}")
    return available


def print_comparison_table(results: dict[str, dict]):
    """Imprime tabla comparativa de todos los modelos."""
    print("\n" + "=" * 75)
    print("  BENCHMARK MULTI-MODELO — TABLA COMPARATIVA")
    print("=" * 75)
    print(f"  {'Modelo':<35} {'Victorias':>9} {'Avg(s)':>7} {'Max(s)':>7} {'Fallback':>9}")
    print("-" * 75)

    # Ordenar por victorias desc, luego por velocidad
    sorted_models = sorted(
        results.items(),
        key=lambda x: (-x[1].get('win_rate', 0), x[1].get('avg_decision_time', 999))
    )

    for model, r in sorted_models:
        win_pct = f"{r.get('win_rate', 0)*100:.0f}% ({r.get('wins', 0)}/{r.get('games', 0)})"
        avg = f"{r.get('avg_decision_time', 0):.2f}s"
        max_t = f"{r.get('max_decision_time', 0):.2f}s"
        fb = f"{r.get('fallback_rate', 0)*100:.1f}%"
        name = model[:35]
        print(f"  {name:<35} {win_pct:>9} {avg:>7} {max_t:>7} {fb:>9}")

    print("=" * 75)
    print()

    # Ganador
    if sorted_models:
        best = sorted_models[0]
        print(f"  🏆 Mejor modelo: {best[0]}")
        print(f"     Win rate: {best[1].get('win_rate', 0)*100:.0f}%")
        print(f"     Avg decisión: {best[1].get('avg_decision_time', 0):.2f}s")
    print()


def main():
    parser = argparse.ArgumentParser(description='CatanLLM Multi-Model Benchmark')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Modelos a comparar (nombres Ollama). Default: lista automática.')
    parser.add_argument('--games', type=int, default=2,
                        help='Partidas por modelo (default: 2)')
    parser.add_argument('--rounds', type=int, default=80,
                        help='Rondas máximas por partida (default: 80)')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout por decisión LLM en segundos (default: 60)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434')
    parser.add_argument('--output', type=str, default=None,
                        help='Guardar resultados en JSON (ej: results/benchmark.json)')
    parser.add_argument('--include-dry-run', action='store_true',
                        help='Incluir baseline con agente aleatorio puro')
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("  🎲 CatanLLM — BENCHMARK MULTI-MODELO")
    print("=" * 75)
    print(f"  Partidas por modelo: {args.games}")
    print(f"  Rondas max/partida:  {args.rounds}")
    print(f"  Timeout LLM:         {args.timeout}s")
    print(f"  Ollama:              {args.ollama_url}")
    print()

    # Verificar Ollama
    client = OllamaClient(base_url=args.ollama_url)
    if not client.is_available():
        print("❌ Ollama no disponible. Arrancalo con: ollama serve")
        sys.exit(1)

    # Determinar modelos
    if args.models:
        wanted_models = args.models
    else:
        installed = client.list_models()
        # Auto-detect: usar los instalados
        all_candidates = MODELS_GPU + MODELS_CPU
        wanted_models = []
        for m in all_candidates:
            if any(m in inst or inst in m for inst in installed):
                if m not in wanted_models:
                    wanted_models.append(m)
        if not wanted_models:
            wanted_models = installed  # Usar todos los instalados

    # Filtrar a disponibles
    print("  Modelos a evaluar:")
    available_models = detect_available_models(client, wanted_models)
    for m in available_models:
        print(f"    ✅ {m}")
    print()

    if not available_models:
        print("❌ No hay modelos disponibles. Instala uno con: ollama pull qwen2.5:3b")
        sys.exit(1)

    # Baseline dry-run
    all_results = {}

    if args.include_dry_run:
        print("📊 Baseline: Random Agent (dry-run, sin LLM)...")
        runner = BenchmarkRunner(dry_run=True, verbose=False)
        summary = runner.run_benchmark(n_games=args.games, max_rounds=args.rounds)
        all_results['random_baseline'] = {
            'model': 'random_baseline',
            'games': summary.total_games,
            'wins': summary.llm_wins,
            'win_rate': summary.win_rate,
            'avg_decision_time': 0.0,
            'max_decision_time': 0.0,
            'avg_game_time': summary.avg_game_time,
            'fallback_rate': 1.0,
        }

    # Benchmark por modelo
    total_start = time.time()

    for model in available_models:
        print(f"\n{'─'*75}")
        print(f"  🤖 Evaluando: {model}")
        print(f"{'─'*75}")

        try:
            runner = BenchmarkRunner(
                model=model,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
                dry_run=False,
                verbose=True,
            )
            summary = runner.run_benchmark(n_games=args.games, max_rounds=args.rounds)

            all_results[model] = {
                'model': model,
                'games': summary.total_games,
                'wins': summary.llm_wins,
                'win_rate': summary.win_rate,
                'avg_decision_time': summary.avg_decision_time,
                'max_decision_time': max(
                    (g.max_decision_time for g in summary.games), default=0
                ),
                'avg_game_time': summary.avg_game_time,
                'fallback_rate': sum(g.fallback_rate for g in summary.games) / max(len(summary.games), 1),
            }

        except KeyboardInterrupt:
            print(f"\n⚠️  Interrumpido durante {model}. Mostrando resultados parciales.")
            break
        except Exception as e:
            print(f"\n❌ Error con {model}: {e}")
            all_results[model] = {'model': model, 'error': str(e)}

    # Tabla comparativa final
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    if valid_results:
        print_comparison_table(valid_results)

    total_elapsed = time.time() - total_start
    print(f"  Tiempo total benchmark: {total_elapsed/60:.1f} min")

    # Guardar resultados
    output_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'config': {
            'games_per_model': args.games,
            'max_rounds': args.rounds,
            'timeout': args.timeout,
        },
        'results': all_results,
    }

    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  💾 Resultados guardados en: {args.output}")
    else:
        # Siempre guardar en results/latest.json
        import os
        os.makedirs('results', exist_ok=True)
        out_path = f"results/benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  💾 Resultados guardados en: {out_path}")


if __name__ == '__main__':
    main()
