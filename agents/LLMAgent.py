"""
LLMAgent v2 - Agente de Catan impulsado por LLM en tiempo real.

Mejoras sobre v1:
- StateEncoder v2: terrenos + probabilidades + puertos + posiciones rivales
- Prompts enriquecidos para build, commerce, thief, game_start
- on_trade_offer: heurística inteligente (acepta si recibe más valor)
- on_having_more_than_7: usa LLM para descartar estratégicamente
- on_turn_start: juega KNIGHT antes de dados (como top agents)
- on_turn_end: juega VICTORY_POINT al final
- on_build_phase: nodos ordenados por score de terreno
- on_commerce_phase: ratios reales de comercio + objetivo concreto
- on_moving_thief: opciones enriquecidas con probabilidades de terreno
"""
import sys
import random
import time
import json
import logging

sys.path.insert(0, '/root/.openclaw/workspace/PyCatan')
sys.path.insert(0, '/root/.openclaw/workspace/CatanLLM')

from Classes.Constants import (MaterialConstants, BuildConstants,
                                DevelopmentCardConstants as DCC,
                                HarborConstants, TerrainConstants)
from Classes.Materials import Materials
from Classes.TradeOffer import TradeOffer
from Interfaces.AgentInterface import AgentInterface
from llm.client import OllamaClient
from llm.state_encoder import StateEncoder, DICE_WEIGHT, TERRAIN_RESOURCE
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
        self.dry_run = dry_run
        self.timeout = timeout
        self.players = []

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
        """Fallback: decisión de construcción con heurística de score."""
        res = self.hand.resources

        # Intentar ciudad primero (mayor valor)
        if res.has_more(BuildConstants.CITY):
            valid = board_instance.valid_city_nodes(self.id)
            if valid:
                # Elegir la ciudad en el nodo con mayor score
                best = max(valid, key=lambda n: self.encoder._node_score(board_instance, n))
                return {'building': BuildConstants.CITY, 'node_id': best}

        # Pueblo
        if res.has_more(BuildConstants.TOWN):
            valid = board_instance.valid_town_nodes(self.id)
            if valid:
                best = max(valid, key=lambda n: self.encoder._node_score(board_instance, n))
                return {'building': BuildConstants.TOWN, 'node_id': best}

        # Carretera hacia nodos de alto score
        if res.has_more(BuildConstants.ROAD):
            valid = board_instance.valid_road_nodes(self.id)
            if valid:
                # Priorizar carreteras hacia nodos con puerto o alto score
                def road_value(r):
                    dst = r['finishing_node']
                    score = self.encoder._node_score(board_instance, dst)
                    has_port = board_instance.nodes[dst].get('harbor', HarborConstants.NONE) != HarborConstants.NONE
                    return score + (5 if has_port else 0)
                best_road = max(valid, key=road_value)
                return {'building': BuildConstants.ROAD,
                        'node_id': best_road['starting_node'],
                        'road_to': best_road['finishing_node']}

        # Carta de desarrollo
        if res.has_more(BuildConstants.CARD):
            return {'building': BuildConstants.CARD}

        return None

    def _enemy_terrain_options(self) -> list:
        """
        Retorna lista de terrenos enemigos ordenados por valor de bloqueo:
        [(terrain_id, resource_name, dice, weight, enemy_player_id), ...]
        """
        options = []
        for t in self._board_ref.terrain:
            if t.get('has_thief'):
                continue
            if t['terrain_type'] == TerrainConstants.DESERT:
                continue
            dice = t['probability']
            weight = DICE_WEIGHT.get(dice, 0)
            if weight == 0:
                continue

            rname = TERRAIN_RESOURCE.get(t['terrain_type'], '?')
            # Ver si hay enemigos en este terreno
            enemy_pid = -1
            has_own = False
            for node_id in t.get('contacting_nodes', []):
                player = self._board_ref.nodes[node_id]['player']
                if player == self.id:
                    has_own = True
                    break
                if player != -1:
                    enemy_pid = player

            if has_own or enemy_pid == -1:
                continue  # No bloquear nuestros propios terrenos; solo donde hay enemigos

            options.append((t['id'], rname, dice, weight, enemy_pid))

        # Ordenar por peso desc (terrenos más valiosos primero)
        return sorted(options, key=lambda x: -x[3])

    # ── AgentInterface methods ───────────────────────────────────────────────

    def on_game_start(self, board_instance):
        """Coloca el primer pueblo y carretera."""
        self.board = board_instance
        self._board_ref = board_instance
        self.turn_count += 1

        valid_nodes = board_instance.valid_starting_nodes()

        # Generar scored list para el prompt
        scored = sorted(
            [(nid, self.encoder._node_score(board_instance, nid),
              self.encoder._node_summary(board_instance, nid))
             for nid in valid_nodes],
            key=lambda x: -x[1]
        )

        state_text = f"You are Player {self.id}. This is the START of the game (place your first settlement)."
        prompt = PromptBuilder.game_start(state_text, scored)
        result = self._ask_llm(prompt)

        if result and 'node_id' in result and 'road_to' in result:
            node_id = int(result['node_id'])
            road_to = int(result['road_to'])
            if node_id in valid_nodes:
                possible_roads = board_instance.nodes[node_id]['adjacent']
                if road_to in possible_roads:
                    return node_id, road_to
                else:
                    return node_id, possible_roads[0]

        # Fallback: mejor nodo por score
        self.fallback_count += 1
        if scored:
            best_node = scored[0][0]
            roads = board_instance.nodes[best_node]['adjacent']
            return best_node, roads[0]
        node_id = random.choice(valid_nodes) if valid_nodes else random.randint(0, 53)
        return node_id, random.choice(board_instance.nodes[node_id]['adjacent'])

    def on_turn_start(self):
        """
        Jugar carta de desarrollo antes de tirar dados.
        Prioridad: KNIGHT (como los top agents) para mover ladrón estratégicamente.
        """
        self.turn_count += 1
        dev_hand = self.development_cards_hand.hand if self.development_cards_hand else []
        if not dev_hand:
            return None

        # Jugar KNIGHT primero (mueve el ladrón antes de tirar dados)
        for i, card in enumerate(dev_hand):
            if card.type == DCC.KNIGHT:
                return self.development_cards_hand.select_card(i)

        return None

    def on_turn_end(self):
        """
        Jugar carta al final del turno.
        Prioridad: VICTORY_POINT para sumar VP inmediatamente.
        """
        dev_hand = self.development_cards_hand.hand if self.development_cards_hand else []
        if not dev_hand:
            return None

        # Jugar VICTORY_POINT (suma VP)
        for i, card in enumerate(dev_hand):
            if card.type == DCC.VICTORY_POINT:
                return self.development_cards_hand.select_card(i)

        return None

    def on_build_phase(self, board_instance):
        """Fase de construcción: el LLM decide qué construir con información enriquecida."""
        self.board = board_instance
        self._board_ref = board_instance

        actions = self.encoder.encode_valid_build_actions(board_instance, self.id, self.hand)

        if not any([actions['can_build_town'], actions['can_build_city'],
                    actions['can_build_road'], actions['can_buy_card']]):
            return None

        game_state = self._get_game_state_text()
        prompt = PromptBuilder.build_phase(game_state, actions)
        result = self._ask_llm(prompt)

        if result:
            action = result.get('action', 'none')

            if action == 'none':
                return None

            elif action == 'city' and actions['can_build_city']:
                node_id = result.get('node_id')
                valid = actions['valid_city_nodes']
                if node_id in valid:
                    return {'building': BuildConstants.CITY, 'node_id': node_id}
                elif valid:
                    # Elegir mejor por score
                    best = actions['valid_city_nodes_scored'][0][0] if actions['valid_city_nodes_scored'] else valid[0]
                    return {'building': BuildConstants.CITY, 'node_id': best}

            elif action == 'town' and actions['can_build_town']:
                node_id = result.get('node_id')
                valid = actions['valid_town_nodes']
                if node_id in valid:
                    return {'building': BuildConstants.TOWN, 'node_id': node_id}
                elif valid:
                    best = actions['valid_town_nodes_scored'][0][0] if actions['valid_town_nodes_scored'] else valid[0]
                    return {'building': BuildConstants.TOWN, 'node_id': best}

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

        # Fallback con heurística
        return self._random_build(board_instance)

    def on_commerce_phase(self):
        """Fase de comercio con ratios reales y objetivo concreto."""
        dev_hand = self.development_cards_hand.hand if self.development_cards_hand else []

        # Intentar jugar cartas de progreso (Year of Plenty, Road Building, Monopoly)
        for i, card in enumerate(dev_hand):
            if hasattr(card, 'effect') and card.effect in [
                DCC.YEAR_OF_PLENTY_EFFECT,
                DCC.ROAD_BUILDING_EFFECT,
                DCC.MONOPOLY_EFFECT,
            ]:
                return self.development_cards_hand.select_card(i)

        if self._board_ref is None:
            return None

        # Calcular ratios reales con puertos
        trade_ratios = self.encoder.encode_trade_ratios(self._board_ref, self.id)
        missing_goal = self.encoder._what_is_missing(self.hand.resources)

        # Verificar si tiene sentido comerciar: ¿tiene suficiente de algún material?
        res = self.hand.resources
        amounts = {'Wheat': res.cereal, 'Ore': res.mineral, 'Clay': res.clay, 'Wood': res.wood, 'Wool': res.wool}
        can_trade = any(amounts[mat] >= trade_ratios[mat] for mat in trade_ratios)
        if not can_trade:
            return None

        game_state = self._get_game_state_text()
        prompt = PromptBuilder.commerce_phase(game_state, trade_ratios, missing_goal)
        result = self._ask_llm(prompt)

        if result:
            if result.get('action') == 'skip':
                return None

            gives_mat = result.get('gives')
            receives_mat = result.get('receives')

            if gives_mat is not None and receives_mat is not None:
                gives_mat = int(gives_mat)
                receives_mat = int(receives_mat)

                if 0 <= gives_mat <= 4 and 0 <= receives_mat <= 4 and gives_mat != receives_mat:
                    mat_names = ['Wheat', 'Ore', 'Clay', 'Wood', 'Wool']
                    mat_name = mat_names[gives_mat]
                    required = trade_ratios.get(mat_name, 4)
                    available = amounts.get(mat_name, 0)

                    if available >= required:
                        return {'gives': gives_mat, 'receives': receives_mat}

        # Fallback: comerciar el recurso más abundante que supere su ratio
        for mat_name, ratio in trade_ratios.items():
            mat_id = ['Wheat', 'Ore', 'Clay', 'Wood', 'Wool'].index(mat_name)
            if amounts[mat_name] >= ratio:
                receives = (mat_id + 1) % 5
                return {'gives': mat_id, 'receives': receives}

        return None

    def on_trade_offer(self, board_instance, offer=TradeOffer(), player_id=int):
        """
        Respuesta a ofertas de comercio de otros jugadores.
        Acepta si recibe más valor del que da (como los top agents).
        También considera el estado actual de recursos.
        """
        try:
            # Heurística principal: acepta si gives.has_more(receives)
            # (lo que da el OFERTANTE = lo que yo RECIBO)
            if offer.gives.has_more(offer.receives):
                return True

            # Criterio adicional: acepta si recibo algo que necesito urgentemente
            if self._board_ref is not None:
                missing_goal = self.encoder._what_is_missing(self.hand.resources)
                if missing_goal:
                    # Ver qué recibo
                    r = offer.gives
                    receives_res = {'Wheat': r.cereal, 'Ore': r.mineral,
                                   'Clay': r.clay, 'Wood': r.wood, 'Wool': r.wool}
                    for mat, amt in receives_res.items():
                        if amt > 0 and mat.lower() in missing_goal.lower():
                            return True  # Me dan lo que necesito

            return False
        except Exception:
            return False

    def on_moving_thief(self):
        """Mover el ladrón con información enriquecida de terrenos enemigos."""
        current_thief = 0
        for terrain in self.board.terrain:
            if terrain.get('has_thief'):
                current_thief = terrain['id']
                break

        enemy_options = self._enemy_terrain_options()

        game_state = self._get_game_state_text()
        prompt = PromptBuilder.move_thief(game_state, current_thief, enemy_options)
        result = self._ask_llm(prompt)

        if result:
            terrain = result.get('terrain', 0)
            player = result.get('player', -1)
            if isinstance(terrain, int) and 0 <= terrain <= 18 and terrain != current_thief:
                return {'terrain': terrain, 'player': int(player)}

        # Fallback: ir al mejor terreno enemigo disponible
        self.fallback_count += 1
        if enemy_options:
            best = enemy_options[0]
            return {'terrain': best[0], 'player': best[4]}

        # Fallback total
        terrain = random.randint(0, 18)
        while terrain == current_thief:
            terrain = random.randint(0, 18)
        return {'terrain': terrain, 'player': -1}

    def on_having_more_than_7_materials_when_thief_is_called(self):
        """Descartar cuando se tiene más de 7 recursos. Usa LLM para decidir qué descartar."""
        res = self.hand.resources
        total = res.cereal + res.mineral + res.clay + res.wood + res.wool
        must_discard = total // 2

        if must_discard <= 0:
            return self.hand

        hand_desc = self.encoder.encode_hand(self.hand)
        missing_goal = self.encoder._what_is_missing(res)
        game_state = self._get_game_state_text()
        prompt = PromptBuilder.discard_resources(game_state, hand_desc, must_discard, missing_goal)
        result = self._ask_llm(prompt)

        if result and all(k in result for k in ['wheat', 'ore', 'clay', 'wood', 'wool']):
            try:
                wh  = int(result['wheat'])
                ore = int(result['ore'])
                cl  = int(result['clay'])
                wd  = int(result['wood'])
                wo  = int(result['wool'])

                # Validar: suma correcta y no supera lo que tenemos
                if (wh + ore + cl + wd + wo == must_discard and
                        wh <= res.cereal and ore <= res.mineral and
                        cl <= res.clay and wd <= res.wood and wo <= res.wool):
                    from Classes.Hand import Hand
                    discard_hand = Hand()
                    discard_hand.resources = Materials(wh, ore, cl, wd, wo)
                    return discard_hand
            except (ValueError, TypeError):
                pass

        # Fallback: descartar los más abundantes, conservando lo necesario para el objetivo
        amounts = [res.cereal, res.mineral, res.clay, res.wood, res.wool]
        discards = [0, 0, 0, 0, 0]
        remaining = must_discard

        # Si estamos cerca de ciudad, conservar Wheat y Ore
        if missing_goal and 'city' in missing_goal:
            # Descartar primero Clay, Wood, Wool (los menos útiles para ciudad)
            priority_discard = [2, 3, 4, 0, 1]  # Clay, Wood, Wool, Wheat, Ore
        else:
            # Descartar los más abundantes
            priority_discard = sorted(range(5), key=lambda i: -amounts[i])

        for idx in priority_discard:
            while remaining > 0 and discards[idx] < amounts[idx]:
                discards[idx] += 1
                amounts[idx] -= 1
                remaining -= 1
            if remaining == 0:
                break

        from Classes.Hand import Hand
        discard_hand = Hand()
        discard_hand.resources = Materials(discards[0], discards[1], discards[2], discards[3], discards[4])
        return discard_hand

    def on_monopoly_card_use(self):
        """Elegir material para carta de monopolio."""
        game_state = self._get_game_state_text()
        prompt = PromptBuilder.monopoly_card(game_state, {})
        result = self._ask_llm(prompt)

        if result and 'material' in result:
            mat = int(result['material'])
            if 0 <= mat <= 4:
                return mat

        # Fallback: el material que más tienen los rivales (estimación: el más común)
        # Si no podemos estimarlo, elegir Wheat u Ore (siempre escasos/deseados)
        return MaterialConstants.CEREAL  # Wheat es el más demandado

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
            for r in valid:
                if r['starting_node'] == node_id and r['finishing_node'] == road_to:
                    return {
                        'node_id': node_id, 'road_to': road_to,
                        'node_id_2': node_id_2, 'road_to_2': road_to_2
                    }

        # Fallback: mejores 2 carreteras por score de destino
        def road_value(r):
            dst = r['finishing_node']
            score = self.encoder._node_score(self._board_ref, dst)
            has_port = self._board_ref.nodes[dst].get('harbor', HarborConstants.NONE) != HarborConstants.NONE
            return score + (5 if has_port else 0)

        sorted_roads = sorted(valid, key=road_value, reverse=True)
        if len(sorted_roads) > 1:
            r1, r2 = sorted_roads[0], sorted_roads[1]
            return {
                'node_id': r1['starting_node'], 'road_to': r1['finishing_node'],
                'node_id_2': r2['starting_node'], 'road_to_2': r2['finishing_node'],
            }
        elif sorted_roads:
            r = sorted_roads[0]
            return {'node_id': r['starting_node'], 'road_to': r['finishing_node'],
                    'node_id_2': None, 'road_to_2': None}
        return None

    def on_year_of_plenty_card_use(self):
        """Usar carta de año de abundancia."""
        missing_goal = self.encoder._what_is_missing(self.hand.resources)
        game_state = self._get_game_state_text()
        prompt = PromptBuilder.year_of_plenty_card(game_state, missing_goal)
        result = self._ask_llm(prompt)

        if result and 'material' in result and 'material_2' in result:
            m1 = int(result['material'])
            m2 = int(result['material_2'])
            if 0 <= m1 <= 4 and 0 <= m2 <= 4:
                return {'material': m1, 'material_2': m2}

        # Fallback: lo que más falta para el objetivo
        if 'city' in (missing_goal or ''):
            return {'material': MaterialConstants.CEREAL, 'material_2': MaterialConstants.MINERAL}
        return {'material': MaterialConstants.CEREAL, 'material_2': MaterialConstants.WOOL}

    # ── Métricas ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
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
