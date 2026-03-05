"""
agent_benchmark.py — Benchmark: LLM vs todos los agentes de PyCatan.

Prueba el modelo LLM contra cada agente rival en todas las posiciones (P0-P3).
Para cada combinación, los 3 oponentes son el mismo agente.

Uso:
    python scripts/agent_benchmark.py
    python scripts/agent_benchmark.py --games 2 --rounds 100
    python scripts/agent_benchmark.py --model qwen2.5:7b-instruct-q4_K_M --output results/agent_benchmark.json
"""
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, '/home/vicente/RoadToDevOps/PyCatan')
sys.path.insert(0, '/home/vicente/catan-workspace/CatanLLM')

from benchmark.runner import BenchmarkRunner, create_llm_agent_class
from llm.client import OllamaClient

# Agentes de PyCatan seguros (excluimos CrabisaAgent: crea GameManager interno)
from Agents.RandomAgent import RandomAgent
from Agents.AlexPastorAgent import AlexPastorAgent
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent
from Agents.TristanAgent import TristanAgent
from Agents.GeneticAgent import GeneticAgent
from Agents.SigmaAgent import SigmaAgent
from Agents.EdoAgent import EdoAgent
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent
from Agents.AdrianHerasAgent import AdrianHerasAgent
from Agents.CarlesZaidaAgent import CarlesZaidaAgent

RIVAL_AGENTS = {
    'RandomAgent': RandomAgent,
    'AlexPastorAgent': AlexPastorAgent,
    'AlexPelochoJaimeAgent': AlexPelochoJaimeAgent,
    'TristanAgent': TristanAgent,
    'GeneticAgent': GeneticAgent,
    'SigmaAgent': SigmaAgent,
    'EdoAgent': EdoAgent,
    'PabloAleixAlexAgent': PabloAleixAlexAgent,
    'AdrianHerasAgent': AdrianHerasAgent,
    'CarlesZaidaAgent': CarlesZaidaAgent,
}


def build_agents_list(llm_class, rival_class, llm_position: int) -> list:
    """Crea lista de 4 agentes con LLM en la posicion indicada."""
    agents = [rival_class] * 4
    agents[llm_position] = llm_class
    return agents


def print_results_table(results: list[dict]):
    """Imprime tabla comparativa final."""
    print("\n" + "=" * 95)
    print("  AGENT BENCHMARK — TABLA COMPARATIVA")
    print("=" * 95)
    print(f"  {'Rival':<25} {'Pos':>3} {'Games':>5} {'Wins':>5} {'WinRate':>8} {'AvgTime':>8} {'Fallback':>9}")
    print("-" * 95)

    for r in results:
        win_pct = f"{r['win_rate']*100:.0f}%"
        avg_t = f"{r['avg_decision_time']:.2f}s"
        fb = f"{r['fallback_rate']*100:.1f}%"
        print(f"  {r['rival']:<25} P{r['llm_position']:>1}   {r['games']:>5} {r['wins']:>5} {win_pct:>8} {avg_t:>8} {fb:>9}")

    print("=" * 95)

    # Resumen por rival (agregado de todas las posiciones)
    rivals = sorted(set(r['rival'] for r in results))
    print(f"\n  {'Rival':<25} {'Total Games':>11} {'Total Wins':>11} {'WinRate':>8}")
    print("-" * 65)
    for rival in rivals:
        rival_results = [r for r in results if r['rival'] == rival]
        total_games = sum(r['games'] for r in rival_results)
        total_wins = sum(r['wins'] for r in rival_results)
        rate = total_wins / total_games if total_games > 0 else 0
        print(f"  {rival:<25} {total_games:>11} {total_wins:>11} {rate*100:>7.0f}%")
    print("=" * 65)
    print()


def main():
    parser = argparse.ArgumentParser(description='CatanLLM Agent Benchmark')
    parser.add_argument('--model', type=str, default='qwen2.5:7b-instruct-q4_K_M',
                        help='Modelo LLM (default: qwen2.5:7b-instruct-q4_K_M)')
    parser.add_argument('--games', type=int, default=2,
                        help='Partidas por combinacion (default: 2)')
    parser.add_argument('--rounds', type=int, default=100,
                        help='Rondas max por partida (default: 100)')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout por decision LLM en segundos (default: 60)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434')
    parser.add_argument('--output', type=str, default=None,
                        help='Guardar resultados en JSON')
    parser.add_argument('--agents', nargs='+', default=None,
                        help='Agentes rivales a probar (default: todos)')
    parser.add_argument('--positions', nargs='+', type=int, default=None,
                        help='Posiciones del LLM a probar, ej: 0 2 (default: 0 1 2 3)')
    args = parser.parse_args()

    # Filtrar agentes
    if args.agents:
        agents_to_test = {k: v for k, v in RIVAL_AGENTS.items() if k in args.agents}
        if not agents_to_test:
            print(f"No se encontraron agentes: {args.agents}")
            print(f"Disponibles: {list(RIVAL_AGENTS.keys())}")
            sys.exit(1)
    else:
        agents_to_test = RIVAL_AGENTS

    positions = args.positions if args.positions else [0, 1, 2, 3]

    total_combos = len(agents_to_test) * len(positions)
    total_games = total_combos * args.games

    print("\n" + "=" * 75)
    print("  CatanLLM — AGENT BENCHMARK")
    print("=" * 75)
    print(f"  Modelo LLM:       {args.model}")
    print(f"  Agentes rivales:  {len(agents_to_test)}")
    print(f"  Posiciones LLM:   {positions}")
    print(f"  Partidas/combo:   {args.games}")
    print(f"  Rondas max:       {args.rounds}")
    print(f"  Total partidas:   {total_games}")
    print(f"  Timeout LLM:      {args.timeout}s")
    print()

    # Verificar Ollama
    client = OllamaClient(base_url=args.ollama_url)
    if not client.is_available():
        print("Ollama no disponible. Arrancalo con: ollama serve")
        sys.exit(1)

    # Runner
    runner = BenchmarkRunner(
        model=args.model,
        ollama_url=args.ollama_url,
        timeout=args.timeout,
        dry_run=False,
        verbose=True,
    )

    LLMAgentClass = create_llm_agent_class(
        model=args.model,
        ollama_url=args.ollama_url,
        timeout=args.timeout,
    )

    all_results = []
    game_counter = 0
    total_start = time.time()

    for rival_name, rival_class in agents_to_test.items():
        for llm_pos in positions:
            combo_wins = 0
            combo_times = []
            combo_fallbacks = []

            print(f"\n{'─'*75}")
            print(f"  LLM(P{llm_pos}) vs 3x {rival_name}")
            print(f"{'─'*75}")

            agents_list = build_agents_list(LLMAgentClass, rival_class, llm_pos)

            for g in range(1, args.games + 1):
                game_counter += 1
                try:
                    metrics, _ = runner.run_single_game_custom(
                        agents_classes=agents_list,
                        llm_position=llm_pos,
                        game_id=game_counter,
                        max_rounds=args.rounds,
                    )
                    if metrics.llm_won:
                        combo_wins += 1
                    combo_times.extend(metrics.decision_times)
                    combo_fallbacks.append(metrics.fallback_rate)

                except KeyboardInterrupt:
                    print("\nInterrumpido. Mostrando resultados parciales.")
                    if all_results:
                        print_results_table(all_results)
                    sys.exit(0)
                except Exception as e:
                    print(f"  Error en partida: {e}")
                    import traceback
                    traceback.print_exc()

            result = {
                'rival': rival_name,
                'llm_position': llm_pos,
                'games': args.games,
                'wins': combo_wins,
                'win_rate': combo_wins / args.games if args.games > 0 else 0,
                'avg_decision_time': sum(combo_times) / len(combo_times) if combo_times else 0,
                'fallback_rate': sum(combo_fallbacks) / len(combo_fallbacks) if combo_fallbacks else 0,
            }
            all_results.append(result)

    total_elapsed = time.time() - total_start

    # Tabla final
    if all_results:
        print_results_table(all_results)

    print(f"  Tiempo total: {total_elapsed/60:.1f} min ({game_counter} partidas)")

    # Guardar JSON
    output_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'config': {
            'model': args.model,
            'games_per_combo': args.games,
            'max_rounds': args.rounds,
            'timeout': args.timeout,
            'positions': positions,
            'agents': list(agents_to_test.keys()),
        },
        'results': all_results,
    }

    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  Resultados guardados en: {args.output}")
    else:
        import os
        os.makedirs('results', exist_ok=True)
        out_path = f"results/agent_benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  Resultados guardados en: {out_path}")


if __name__ == '__main__':
    main()
