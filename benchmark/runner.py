"""
BenchmarkRunner - Ejecuta partidas de Catan con LLMAgent y mide rendimiento.

Nota: PyCatan espera CLASES (no instancias) en el parámetro `agents`.
Usamos el patrón factory para crear clases configuradas con los parámetros del LLM.
"""
import sys
import time
import logging

sys.path.insert(0, '/root/.openclaw/workspace/PyCatan')
sys.path.insert(0, '/root/.openclaw/workspace/CatanLLM')

from Managers.GameDirector import GameDirector
from Agents.RandomAgent import RandomAgent
from agents.LLMAgent import LLMAgent
from llm.client import OllamaClient
from benchmark.metrics import GameMetrics, BenchmarkSummary

logger = logging.getLogger(__name__)


def create_llm_agent_class(model: str = 'qwen2.5:3b',
                           ollama_url: str = 'http://localhost:11434',
                           timeout: int = 30,
                           dry_run: bool = False):
    """
    Factory: devuelve una CLASE de LLMAgent ya configurada.
    PyCatan llama a agents[0](agent_id) para crear instancias, así que la clase
    debe tener solo agent_id como parámetro.
    """
    class ConfiguredLLMAgent(LLMAgent):
        def __init__(self, agent_id: int):
            super().__init__(
                agent_id=agent_id,
                model=model,
                ollama_url=ollama_url,
                timeout=timeout,
                dry_run=dry_run,
            )
    ConfiguredLLMAgent.__name__ = f'LLMAgent({model})'
    return ConfiguredLLMAgent


class BenchmarkRunner:
    """
    Ejecuta partidas de Catan donde un LLMAgent juega contra RandomAgents.
    Mide tiempo de decisión, tasa de éxito del LLM, victorias.
    """

    def __init__(self, model: str = 'qwen2.5:3b', ollama_url: str = 'http://localhost:11434',
                 timeout: int = 30, dry_run: bool = False, verbose: bool = True):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.dry_run = dry_run
        self.verbose = verbose

        # Verificar conexión con Ollama si no es dry_run
        if not dry_run:
            client = OllamaClient(model=model, base_url=ollama_url)
            if not client.is_available():
                print("⚠️  Ollama no está disponible en localhost:11434")
                print("   Para arrancarlo: ollama serve &")
                print("   Para instalar:   bash scripts/install_model.sh")
            else:
                models = client.list_models()
                if models:
                    print(f"✅ Ollama disponible. Modelos instalados: {models}")
                    if not any(model in m for m in models):
                        print(f"⚠️  El modelo '{model}' no está descargado.")
                        print(f"   Ejecuta: ollama pull {model}")
                else:
                    print("⚠️  Ollama disponible pero sin modelos.")
                    print("   Ejecuta: ollama pull qwen2.5:3b")

    def run_single_game(self, game_id: int = 1, max_rounds: int = 500,
                        print_outcome: bool = False) -> tuple[GameMetrics, LLMAgent]:
        """Ejecuta una partida: 1 LLMAgent (P0) vs 3 RandomAgents."""

        # Crear la clase configurada de LLMAgent
        LLMAgentClass = create_llm_agent_class(
            model=self.model,
            ollama_url=self.ollama_url,
            timeout=self.timeout,
            dry_run=self.dry_run,
        )

        # GameDirector espera clases, no instancias
        agents_classes = [LLMAgentClass, RandomAgent, RandomAgent, RandomAgent]

        gd = GameDirector(
            agents=agents_classes,
            max_rounds=max_rounds,
            store_trace=False,
        )

        metrics = GameMetrics(
            game_id=game_id,
            llm_player_id=0,
            model_name=self.model if not self.dry_run else 'dry_run',
        )

        if self.verbose:
            mode = 'DRY RUN (random fallback)' if self.dry_run else f'LLM ({self.model})'
            print(f"\n🎲 Partida {game_id} | P0={mode} vs P1,P2,P3=Random | MaxRounds={max_rounds}")

        start_time = time.time()

        # Jugar (game_start hace todo el loop interno)
        try:
            gd.game_start(game_number=game_id, print_outcome=print_outcome)
        except Exception as e:
            logger.error(f"Error en partida {game_id}: {e}")
            if self.verbose:
                import traceback
                print(f"❌ Error en partida: {e}")
                traceback.print_exc()

        total_time = time.time() - start_time

        # Extraer la instancia del LLMAgent creada internamente por PyCatan
        llm_instance: LLMAgent = gd.game_manager.agent_manager.players[0]['player']

        # Determinar ganador
        winner_id = -1
        players = gd.game_manager.agent_manager.players
        for p in players:
            if p.get('victory_points', 0) >= 10:
                winner_id = p['id']
                break
        # Si no hubo ganador claro (max_rounds alcanzado), buscar el que más puntos tiene
        if winner_id == -1:
            winner_id = max(players, key=lambda p: p.get('victory_points', 0))['id']

        # Registrar métricas
        llm_metrics = llm_instance.get_metrics()
        metrics.llm_decisions = llm_metrics['llm_calls']
        metrics.fallback_decisions = llm_metrics['fallback_count']
        metrics.decision_times = list(llm_instance.decision_times)
        metrics.total_turns = llm_instance.turn_count
        metrics.finish(winner_id)

        if self.verbose:
            metrics.print_summary()
            if not self.dry_run:
                llm_stats = llm_instance.llm.stats() if llm_instance.llm else {}
                if llm_stats:
                    print(f"  LLM stats: {llm_stats}")

        return metrics, llm_instance

    def run_benchmark(self, n_games: int = 5, max_rounds: int = 500) -> BenchmarkSummary:
        """Ejecuta N partidas y genera resumen comparativo."""
        print(f"\n{'='*60}")
        mode = f'LLM: {self.model}' if not self.dry_run else 'DRY RUN'
        print(f"  BENCHMARK: {n_games} partidas | {mode}")
        print(f"{'='*60}")

        all_metrics = []
        for i in range(1, n_games + 1):
            metrics, _ = self.run_single_game(game_id=i, max_rounds=max_rounds)
            all_metrics.append(metrics)

        summary = BenchmarkSummary(all_metrics)
        summary.print_summary()
        return summary
