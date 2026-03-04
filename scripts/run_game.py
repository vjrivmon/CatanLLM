"""
run_game.py - Corre una partida de Catan con el LLMAgent y muestra métricas.

Uso:
    python scripts/run_game.py                          # LLM vs Random (1 partida)
    python scripts/run_game.py --dry-run                # Sin LLM (solo mide lógica)
    python scripts/run_game.py --model qwen2.5:7b       # Usar modelo específico
    python scripts/run_game.py --games 3                # Correr 3 partidas
    python scripts/run_game.py --max-rounds 300         # Limitar rondas por partida
"""
import sys
import argparse
import logging

# Añadir paths
sys.path.insert(0, '/root/.openclaw/workspace/PyCatan')
sys.path.insert(0, '/root/.openclaw/workspace/CatanLLM')


def main():
    parser = argparse.ArgumentParser(description='CatanLLM - LLM juega Catan en tiempo real')
    parser.add_argument('--model', type=str, default='qwen2.5:3b',
                        help='Modelo Ollama a usar (default: qwen2.5:3b)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                        help='URL del servidor Ollama (default: http://localhost:11434)')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout por decisión LLM en segundos (default: 30)')
    parser.add_argument('--dry-run', action='store_true',
                        help='No usar LLM, solo fallback aleatorio (para tests de la integración)')
    parser.add_argument('--games', type=int, default=1,
                        help='Número de partidas a jugar (default: 1)')
    parser.add_argument('--max-rounds', type=int, default=500,
                        help='Rondas máximas por partida (default: 500)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Mostrar output detallado')
    parser.add_argument('--log-level', type=str, default='WARNING',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Nivel de logging (default: WARNING)')
    args = parser.parse_args()

    # Configurar logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # Importar después de configurar paths
    from benchmark.runner import BenchmarkRunner
    from llm.client import OllamaClient

    print("\n" + "="*60)
    print("  🎲 CATANLLM - LLM plays Catan in real-time")
    print("="*60)

    if args.dry_run:
        print("  Mode: DRY RUN (no LLM calls, using random fallback)")
    else:
        print(f"  Model:   {args.model}")
        print(f"  Ollama:  {args.ollama_url}")
        print(f"  Timeout: {args.timeout}s per decision")

        # Verificar estado de Ollama
        client = OllamaClient(model=args.model, base_url=args.ollama_url)
        if not client.is_available():
            print("\n⚠️  Ollama no está disponible!")
            print("  Opciones:")
            print("  1. Arrancarlo: ollama serve &")
            print("  2. Instalarlo: bash scripts/install_model.sh")
            print("  3. Correr en dry-run: python scripts/run_game.py --dry-run")
            sys.exit(1)

        models = client.list_models()
        model_available = any(args.model in m for m in models)
        if not model_available:
            print(f"\n⚠️  Modelo '{args.model}' no encontrado.")
            print(f"  Modelos disponibles: {models}")
            print(f"  Para bajarlo: ollama pull {args.model}")
            if models:
                print(f"\n  ¿Usar '{models[0]}' en su lugar? Pasa --model {models[0]}")
            sys.exit(1)

    print(f"  Games:   {args.games}")
    print(f"  Rounds:  max {args.max_rounds} per game")
    print("="*60)

    runner = BenchmarkRunner(
        model=args.model,
        ollama_url=args.ollama_url,
        timeout=args.timeout,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    if args.games == 1:
        metrics, agent = runner.run_single_game(game_id=1, max_rounds=args.max_rounds)
    else:
        summary = runner.run_benchmark(n_games=args.games, max_rounds=args.max_rounds)


if __name__ == '__main__':
    main()
