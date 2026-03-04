"""
LLMAgent - Agente de Catan impulsado por LLM en tiempo real.

Cada turno: recibe estado del juego como texto → consulta al LLM → ejecuta acción.
Si el LLM falla/timeout → fallback a comportamiento aleatorio.
"""
import sys
import random
import time
import json
import logging

sys.path.insert(0, '/root/.openclaw/workspace/PyCatan')
sys.path.insert(0, '/root/.openclaw/workspace/CatanLLM')

from Classes.Constants import MaterialConstants, BuildConstants
from Classes.Materials import Materials
from Classes.TradeOffer import TradeOffer
from Interfaces.AgentInterface import AgentInterface
from llm.client import OllamaClient
from llm.state_encoder import StateEncoder
from llm.prompts import PromptBuilder

logger = logging.getLogger(__name__)


class LLMAgent(AgentInterface):
    """
    Agente de Catan que usa un LLM local (via Ollama) para tomar decisiones.
    
    Métricas capturadas:
    - Tiempo por decisión
    - Número de fallbacks por timeout/error
    - Total de consultas LLM
    """

    def __init__(self, agent_id: int, model: str = 'qwen2.5:3b',
                 ollama_url: str = 'http://localhost:11434',
                 timeout: int = 30, dry_run: bool = False):
        super().__init__(agent_id)
        self.model = model
        self.dry_run = dry_run  # Si True, usa solo fallback aleatorio (para tests sin LLM)
        self.timeout = timeout
        self.players = []  # Se actualiza externamente

        # LLM client y helpers
        if not dry_run:
            self.llm = OllamaClient(model=model, base_url=ollama_url)
        else:
            self.llm = None

        self.encoder = StateEncoder()
        self.prompts = PromptBuilder()

        # Métricas
        self.decision_times = []
        self.fallback_count = 0
        self.llm_success_count = 0
        self.turn_count = 0

        # Estado disponible para LLM
        self._board_ref = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_game_state_text(self) -> str:
        if self._board_ref is None:
            return "State unavailable"
        return self.encoder.encode_full_state(
            board=self._board_ref,
            hand=self.hand,
            dev_cards_hand=self.development_cards_hand,
            player_id=self.id,
            players=self.players,
        )

    def _ask_llm(self, prompt: str) -> dict | None:
        """Consulta al LLM y retorna JSON parseado, o None si falla."""
        if self.dry_run or self.llm is None:
            return None

        try:
            start = time.time()
            response = self.llm.generate(prompt, timeout=self.timeout)
            elapsed = time.time() - start
            self.decision_times.append(elapsed)

            parsed = PromptBuilder.parse_json_response(response)
            if parsed is not None:
                self.llm_success_count += 1
                logger.debug(f"[LLM P{self.id}] {elapsed:.2f}s → {parsed}")
                return parsed
            else:
                logger.warning(f"[LLM P{self.id}] No se pudo parsear JSON: {response[:100]}")
                self.fallback_count += 1
                return None

        except (TimeoutError, ConnectionError) as e:
            logger.warning(f"[LLM P{self.id}] Error: {e} → fallback")
            self.fallback_count += 1
            return None
        except Exception as e:
            logger.warning(f"[LLM P{self.id}] Error inesperado: {e} → fallback")
            self.fallback_count += 1
            return None

    def _random_build(self, board_instance):
        """Fallback: decisión de construcción aleatoria (como RandomAgent)."""
        res = self.hand.resources

        answer = random.randint(0, 2)
        if res.has_more(BuildConstants.TOWN) and answer == 0:
            if random.randint(0, 1):
                valid = board_instance.valid_town_nodes(self.id)
                if valid:
                    return {'building': BuildConstants.TOWN, 'node_id': random.choice(valid)}
            else:
                valid = board_instance.valid_road_nodes(self.id)
                if valid:
                    r = random.choice(valid)
                    return {'building': BuildConstants.ROAD, 'node_id': r['starting_node'], 'road_to': r['finishing_node']}

        elif res.has_more(BuildConstants.CITY) and answer == 1:
            valid = board_instance.valid_city_nodes(self.id)
            if valid:
                return {'building': BuildConstants.CITY, 'node_id': random.choice(valid)}

        elif res.has_more(BuildConstants.CARD) and answer == 2:
            return {'building': BuildConstants.CARD}

        return None

    # ── AgentInterface methods ───────────────────────────────────────────────

    def on_game_start(self, board_instance):
        """Coloca el primer pueblo y carretera."""
        self.board = board_instance
        self._board_ref = board_instance
        self.turn_count += 1

        # Intentar con LLM
        valid_nodes = list(range(54))
        state_text = f"You are Player {self.id}. This is the start of the game."
        prompt = PromptBuilder.game_start(state_text, valid_nodes)
        result = self._ask_llm(prompt)

        if result and 'node_id' in result and 'road_to' in result:
            node_id = int(result['node_id'])
            road_to = int(result['road_to'])
            # Validar que el nodo y la carretera existen
            if 0 <= node_id <= 53:
                possible_roads = board_instance.nodes[node_id]['adjacent']
                if road_to in possible_roads:
                    return node_id, road_to
                else:
                    return node_id, possible_roads[0]

        # Fallback aleatorio
        self.fallback_count += 1
        node_id = random.randint(0, 53)
        possible_roads = board_instance.nodes[node_id]['adjacent']
        return node_id, random.choice(possible_roads)

    def on_turn_start(self):
        """Jugar carta de desarrollo antes de tirar dados."""
        self.turn_count += 1
        if len(self.development_cards_hand.hand) and random.randint(0, 3) == 0:
            return self.development_cards_hand.select_card(0)
        return None

    def on_turn_end(self):
        """Jugar carta de desarrollo al final del turno."""
        if len(self.development_cards_hand.hand) and random.randint(0, 3) == 0:
            return self.development_cards_hand.select_card(0)
        return None

    def on_build_phase(self, board_instance):
        """Fase de construcción: el LLM decide qué construir."""
        self.board = board_instance
        self._board_ref = board_instance

        # Obtener acciones válidas
        actions = self.encoder.encode_valid_build_actions(board_instance, self.id, self.hand)

        # Si no hay nada que hacer, salir
        if not any([actions['can_build_town'], actions['can_build_city'],
                    actions['can_build_road'], actions['can_buy_card']]):
            return None

        # Preguntar al LLM
        game_state = self._get_game_state_text()
        prompt = PromptBuilder.build_phase(game_state, actions)
        result = self._ask_llm(prompt)

        if result:
            action = result.get('action', 'none')

            if action == 'none':
                return None

            elif action == 'town' and actions['can_build_town']:
                node_id = result.get('node_id')
                valid = actions['valid_town_nodes']
                if node_id in valid:
                    return {'building': BuildConstants.TOWN, 'node_id': node_id}
                elif valid:
                    return {'building': BuildConstants.TOWN, 'node_id': valid[0]}

            elif action == 'city' and actions['can_build_city']:
                node_id = result.get('node_id')
                valid = actions['valid_city_nodes']
                if node_id in valid:
                    return {'building': BuildConstants.CITY, 'node_id': node_id}
                elif valid:
                    return {'building': BuildConstants.CITY, 'node_id': valid[0]}

            elif action == 'road' and actions['can_build_road']:
                node_id = result.get('node_id')
                road_to = result.get('road_to')
                valid = actions['valid_road_nodes']
                for r in valid:
                    if r['starting_node'] == node_id and r['finishing_node'] == road_to:
                        return {'building': BuildConstants.ROAD, 'node_id': node_id, 'road_to': road_to}
                if valid:
                    r = valid[0]
                    return {'building': BuildConstants.ROAD, 'node_id': r['starting_node'], 'road_to': r['finishing_node']}

            elif action == 'card' and actions['can_buy_card']:
                return {'building': BuildConstants.CARD}

        # Fallback aleatorio
        return self._random_build(board_instance)

    def on_commerce_phase(self):
        """Fase de comercio: el LLM decide si comerciar."""
        if len(self.development_cards_hand.hand) and random.randint(0, 1):
            return self.development_cards_hand.select_card(0)

        game_state = self._get_game_state_text()
        prompt = PromptBuilder.commerce_phase(game_state)
        result = self._ask_llm(prompt)

        if result:
            if result.get('action') == 'skip':
                return None
            gives = result.get('gives')
            receives = result.get('receives')
            if gives is not None and receives is not None:
                gives = int(gives)
                receives = int(receives)
                if 0 <= gives <= 4 and 0 <= receives <= 4:
                    # Comprobar si podemos permitirnos dar 4
                    res = self.hand.resources
                    amounts = [res.cereal, res.mineral, res.clay, res.wood, res.wool]
                    if amounts[gives] >= 4:
                        return {'gives': gives, 'receives': receives}

        # Fallback: comerciar si tenemos 4+ de algo
        res = self.hand.resources
        for mat, amount in [(MaterialConstants.CEREAL, res.cereal),
                            (MaterialConstants.MINERAL, res.mineral),
                            (MaterialConstants.CLAY, res.clay),
                            (MaterialConstants.WOOD, res.wood),
                            (MaterialConstants.WOOL, res.wool)]:
            if amount >= 4:
                receives = (mat + 1) % 5
                return {'gives': mat, 'receives': receives}

        return None

    def on_trade_offer(self, board_instance, offer=TradeOffer(), player_id=int):
        """Respuesta a ofertas de comercio (siempre rechaza por simplicidad)."""
        return False

    def on_moving_thief(self):
        """Mover el ladrón."""
        # Encontrar terreno actual del ladrón
        current_thief = 0
        for terrain in self.board.terrain:
            if terrain.get('has_thief'):
                current_thief = terrain['id']
                break

        game_state = self._get_game_state_text()
        prompt = PromptBuilder.move_thief(game_state, current_thief)
        result = self._ask_llm(prompt)

        if result:
            terrain = result.get('terrain', 0)
            player = result.get('player', -1)
            if isinstance(terrain, int) and 0 <= terrain <= 18 and terrain != current_thief:
                return {'terrain': terrain, 'player': int(player)}

        # Fallback: mover al ladrón aleatoriamente
        terrain = random.randint(0, 18)
        while terrain == current_thief:
            terrain = random.randint(0, 18)

        player = -1
        for node in self.board.terrain[terrain].get('contacting_nodes', []):
            if self.board.nodes[node]['player'] != -1 and self.board.nodes[node]['player'] != self.id:
                player = self.board.nodes[node]['player']
                break
        return {'terrain': terrain, 'player': player}

    def on_having_more_than_7_materials_when_thief_is_called(self):
        """Descartar cuando se tiene más de 7 recursos."""
        res = self.hand.resources
        total = res.cereal + res.mineral + res.clay + res.wood + res.wool
        must_discard = total // 2

        if must_discard <= 0:
            return self.hand

        # Fallback: descartar los recursos más abundantes (estrategia simple)
        amounts = [res.cereal, res.mineral, res.clay, res.wood, res.wool]
        discards = [0, 0, 0, 0, 0]
        remaining = must_discard

        # Descartar en orden de abundancia
        while remaining > 0:
            max_idx = amounts.index(max(amounts))
            if amounts[max_idx] <= 0:
                break
            discards[max_idx] += 1
            amounts[max_idx] -= 1
            remaining -= 1

        from Classes.Hand import Hand
        from Classes.Materials import Materials
        discard_hand = Hand()
        discard_hand.resources = Materials(discards[0], discards[1], discards[2], discards[3], discards[4])
        return discard_hand

    def on_monopoly_card_use(self):
        """Elegir material para carta de monopolio."""
        game_state = self._get_game_state_text()
        prompt = PromptBuilder.monopoly_card(game_state)
        result = self._ask_llm(prompt)

        if result and 'material' in result:
            mat = int(result['material'])
            if 0 <= mat <= 4:
                return mat

        return random.randint(0, 4)

    def on_road_building_card_use(self):
        """Usar carta de construcción de carreteras."""
        if self._board_ref is None:
            return None

        valid = self._board_ref.valid_road_nodes(self.id)
        if not valid:
            return None

        game_state = self._get_game_state_text()
        prompt = PromptBuilder.road_building_card(game_state, valid)
        result = self._ask_llm(prompt)

        if result and 'node_id' in result and 'road_to' in result:
            node_id = result.get('node_id')
            road_to = result.get('road_to')
            node_id_2 = result.get('node_id_2')
            road_to_2 = result.get('road_to_2')
            # Validar
            for r in valid:
                if r['starting_node'] == node_id and r['finishing_node'] == road_to:
                    return {
                        'node_id': node_id, 'road_to': road_to,
                        'node_id_2': node_id_2, 'road_to_2': road_to_2
                    }

        # Fallback aleatorio
        if len(valid) > 1:
            r1, r2 = random.sample(valid, 2)
            return {
                'node_id': r1['starting_node'], 'road_to': r1['finishing_node'],
                'node_id_2': r2['starting_node'], 'road_to_2': r2['finishing_node'],
            }
        elif valid:
            r = valid[0]
            return {'node_id': r['starting_node'], 'road_to': r['finishing_node'],
                    'node_id_2': None, 'road_to_2': None}
        return None

    def on_year_of_plenty_card_use(self):
        """Usar carta de año de abundancia."""
        game_state = self._get_game_state_text()
        prompt = PromptBuilder.year_of_plenty_card(game_state)
        result = self._ask_llm(prompt)

        if result and 'material' in result and 'material_2' in result:
            m1 = int(result['material'])
            m2 = int(result['material_2'])
            if 0 <= m1 <= 4 and 0 <= m2 <= 4:
                return {'material': m1, 'material_2': m2}

        return {'material': random.randint(0, 4), 'material_2': random.randint(0, 4)}

    # ── Métricas ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Retorna métricas de rendimiento del agente."""
        times = self.decision_times
        llm_stats = self.llm.stats() if self.llm else {}
        return {
            'player_id': self.id,
            'model': self.model,
            'dry_run': self.dry_run,
            'turn_count': self.turn_count,
            'llm_calls': self.llm_success_count,
            'fallback_count': self.fallback_count,
            'avg_decision_time_s': round(sum(times) / len(times), 3) if times else 0,
            'max_decision_time_s': round(max(times), 3) if times else 0,
            'min_decision_time_s': round(min(times), 3) if times else 0,
            'total_decision_time_s': round(sum(times), 2),
            **llm_stats,
        }
