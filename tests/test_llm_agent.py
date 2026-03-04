"""
Tests básicos del LLMAgent y sus componentes.
Usan dry_run=True para no necesitar Ollama activo.
"""
import sys
import unittest

sys.path.insert(0, '/root/.openclaw/workspace/PyCatan')
sys.path.insert(0, '/root/.openclaw/workspace/CatanLLM')


class TestPromptParser(unittest.TestCase):
    """Tests del parser de JSON en respuestas LLM."""

    def setUp(self):
        from llm.prompts import PromptBuilder
        self.pb = PromptBuilder

    def test_clean_json(self):
        result = self.pb.parse_json_response('{"action": "road", "node_id": 5, "road_to": 6}')
        self.assertEqual(result['action'], 'road')
        self.assertEqual(result['node_id'], 5)

    def test_json_with_explanation(self):
        result = self.pb.parse_json_response(
            'I will build a road. {"action": "road", "node_id": 5, "road_to": 6} This is strategic.'
        )
        self.assertIsNotNone(result)
        self.assertEqual(result['action'], 'road')

    def test_json_in_markdown(self):
        result = self.pb.parse_json_response(
            '```json\n{"action": "town", "node_id": 12, "road_to": null}\n```'
        )
        self.assertIsNotNone(result)
        self.assertEqual(result['action'], 'town')

    def test_invalid_json(self):
        result = self.pb.parse_json_response('I cannot decide what to do here.')
        self.assertIsNone(result)


class TestStateEncoder(unittest.TestCase):
    """Tests del encoder de estado del juego."""

    def setUp(self):
        from llm.state_encoder import StateEncoder
        self.encoder = StateEncoder()

    def test_encode_hand(self):
        from Classes.Hand import Hand
        from Classes.Materials import Materials
        hand = Hand()
        hand.resources = Materials(2, 1, 0, 3, 1)
        result = self.encoder.encode_hand(hand)
        self.assertIn('Wheat=2', result)
        self.assertIn('Ore=1', result)
        self.assertIn('Wood=3', result)


class TestLLMAgentDryRun(unittest.TestCase):
    """Tests del LLMAgent en modo dry_run (sin Ollama)."""

    def test_agent_creation(self):
        from agents.LLMAgent import LLMAgent
        agent = LLMAgent(agent_id=0, dry_run=True)
        self.assertEqual(agent.id, 0)
        self.assertTrue(agent.dry_run)
        self.assertIsNone(agent.llm)

    def test_metrics_start_empty(self):
        from agents.LLMAgent import LLMAgent
        agent = LLMAgent(agent_id=0, dry_run=True)
        metrics = agent.get_metrics()
        self.assertEqual(metrics['llm_calls'], 0)
        self.assertEqual(metrics['fallback_count'], 0)
        self.assertEqual(metrics['avg_decision_time_s'], 0)

    def test_game_start_dry_run(self):
        """on_game_start debe retornar (node_id, road_to) válidos."""
        from agents.LLMAgent import LLMAgent
        from Classes.Board import Board
        agent = LLMAgent(agent_id=0, dry_run=True)
        board = Board()
        node_id, road_to = agent.on_game_start(board)
        self.assertIsInstance(node_id, int)
        self.assertIsInstance(road_to, int)
        self.assertIn(road_to, board.nodes[node_id]['adjacent'])

    def test_build_phase_dry_run_no_resources(self):
        """Con mano vacía, on_build_phase debe retornar None."""
        from agents.LLMAgent import LLMAgent
        from Classes.Board import Board
        agent = LLMAgent(agent_id=0, dry_run=True)
        board = Board()
        result = agent.on_build_phase(board)
        self.assertIsNone(result)

    def test_commerce_phase_dry_run(self):
        """on_commerce_phase con mano vacía retorna None."""
        from agents.LLMAgent import LLMAgent
        agent = LLMAgent(agent_id=0, dry_run=True)
        result = agent.on_commerce_phase()
        self.assertIsNone(result)

    def test_moving_thief_dry_run(self):
        """on_moving_thief retorna dict válido."""
        from agents.LLMAgent import LLMAgent
        from Classes.Board import Board
        agent = LLMAgent(agent_id=0, dry_run=True)
        board = Board()
        agent.board = board
        agent._board_ref = board
        result = agent.on_moving_thief()
        self.assertIn('terrain', result)
        self.assertIn('player', result)
        self.assertIsInstance(result['terrain'], int)


class TestOllamaClientConnection(unittest.TestCase):
    """Tests del cliente Ollama (solo disponibilidad, no requiere modelo)."""

    def test_client_creation(self):
        from llm.client import OllamaClient
        client = OllamaClient(model='qwen2.5:3b')
        self.assertEqual(client.model, 'qwen2.5:3b')

    def test_is_available_returns_bool(self):
        from llm.client import OllamaClient
        client = OllamaClient()
        result = client.is_available()
        self.assertIsInstance(result, bool)
        # Solo informa, no falla si Ollama no está corriendo
        print(f"\n  Ollama disponible: {result}")

    def test_stats_empty(self):
        from llm.client import OllamaClient
        client = OllamaClient()
        stats = client.stats()
        self.assertEqual(stats['total_calls'], 0)
        self.assertEqual(stats['total_tokens'], 0)


class TestFullDryRunGame(unittest.TestCase):
    """Test de integración completo: una partida en dry_run."""

    def test_dry_run_game_completes(self):
        """Una partida completa en dry_run debe terminar sin errores."""
        from benchmark.runner import BenchmarkRunner
        runner = BenchmarkRunner(dry_run=True, verbose=False)
        metrics, agent = runner.run_single_game(game_id=1, max_rounds=100)
        self.assertIsNotNone(metrics)
        self.assertIsNotNone(agent)
        self.assertGreaterEqual(metrics.total_time_s, 0)
        print(f"\n  Dry run completado: {metrics.total_turns} turnos en {metrics.total_time_s}s")


if __name__ == '__main__':
    print("="*60)
    print("  CatanLLM - Tests")
    print("="*60)
    unittest.main(verbosity=2)
