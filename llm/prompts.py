"""
PromptBuilder - Plantillas de prompts para cada fase del juego Catan.
El LLM debe responder SIEMPRE con JSON válido.

v2: Prompts enriquecidos con terrenos, probabilidades, ratios de comercio y estrategia.
"""
import json


SYSTEM_PREAMBLE = """You are an expert Catan player. Your ONLY goal: reach 10 victory points first.
Strategy priorities (in order):
1. Upgrade towns to CITIES on your highest-score nodes (cities = 2 VP, double production)
2. Build towns on high-score nodes (★★★★★ = best, dice 6 or 8)
3. Trade to get what you need — use your BEST trade ratio for each resource
4. Build roads toward high-value nodes or harbors
5. Buy dev cards when you have nothing better to do

You MUST reply with ONLY valid JSON, no explanation, no markdown, just JSON."""


class PromptBuilder:

    @staticmethod
    def build_phase(game_state: str, actions: dict) -> str:
        """
        Prompt enriquecido para la fase de construcción.
        Incluye scores de terreno por cada nodo candidato.
        """
        # Construir descripción de opciones con scores
        options = []

        if actions['can_build_city']:
            city_info = []
            for (nid, score, summary) in actions.get('valid_city_nodes_scored', []):
                city_info.append(f"  {summary}")
            options.append(f"CITY (2VP, doubles production) — candidates:\n" + "\n".join(city_info))

        if actions['can_build_town']:
            town_info = []
            for (nid, score, summary) in actions.get('valid_town_nodes_scored', []):
                town_info.append(f"  {summary}")
            options.append(f"TOWN (1VP, new production point) — candidates (sorted best first):\n" + "\n".join(town_info))

        if actions['can_build_road']:
            roads = actions.get('valid_road_nodes', [])
            road_strs = [f"({r['starting_node']}→{r['finishing_node']})" for r in roads]
            options.append(f"ROAD (expansion) — options (sorted toward best nodes): {road_strs}")

        if actions['can_buy_card']:
            options.append("CARD (dev card — 1Wh+1Ore+1Wo)")

        options.append("NONE (skip build phase)")

        options_text = "\n\n".join(options)

        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== BUILD PHASE ===
Available actions (★ = more likely to produce, higher score = better node):

{options_text}

Pick the BEST strategic action. Prefer city > town > road > card > none.
Choose the highest-score node when building town or city.

Reply with ONLY this JSON:
{{"action": "city|town|road|card|none", "node_id": <integer or null>, "road_to": <integer or null>}}

Examples:
{{"action": "city", "node_id": 12, "road_to": null}}
{{"action": "town", "node_id": 34, "road_to": null}}
{{"action": "road", "node_id": 5, "road_to": 6}}
{{"action": "none", "node_id": null, "road_to": null}}"""

    @staticmethod
    def game_start(board_info: str, valid_nodes_scored: list) -> str:
        """
        Prompt para colocar el primer pueblo.
        valid_nodes_scored: [(node_id, score, summary), ...]
        """
        top = valid_nodes_scored[:12] if len(valid_nodes_scored) > 12 else valid_nodes_scored
        nodes_text = "\n".join(f"  {summary}" for (nid, score, summary) in top)

        return f"""{SYSTEM_PREAMBLE}

=== GAME START - PLACE YOUR FIRST SETTLEMENT ===
{board_info}

Available starting nodes (sorted best first by production score):
{nodes_text}

Strategy: pick the node with highest score (diverse resources + high dice numbers).
Also pick an adjacent road direction that points toward more good terrain.

Reply with ONLY this JSON:
{{"node_id": <integer>, "road_to": <integer>}}"""

    @staticmethod
    def commerce_phase(game_state: str, trade_ratios: dict, missing_goal: str) -> str:
        """
        Prompt enriquecido para comercio.
        trade_ratios: {'Wheat': 4, 'Ore': 2, ...}
        missing_goal: texto de qué falta para la siguiente construcción
        """
        ratios_text = ', '.join(f"{mat}({ratio}:1)" for mat, ratio in trade_ratios.items())
        goal_text = f"Your next goal: {missing_goal}" if missing_goal else "No immediate build goal — save resources."

        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== COMMERCE PHASE ===
{goal_text}

Your trade ratios with the bank: {ratios_text}
(Use your best ratio: give N of same resource → receive 1 of any)

Materials: 0=Wheat, 1=Ore, 2=Clay, 3=Wood, 4=Wool

Should you trade? ONLY trade if it gets you closer to your build goal.
- Give the resource you have most of (especially if ratio ≤ 3)
- Receive the resource you need most for your goal
- Skip if you already have what you need, or have nothing to spare

Reply with ONLY this JSON:
{{"gives": <material_int 0-4>, "receives": <material_int 0-4>}}
Or to skip: {{"action": "skip"}}"""

    @staticmethod
    def trade_offer(game_state: str, offer_desc: str) -> str:
        """
        Prompt para responder a oferta de comercio de otro jugador.
        offer_desc: texto describiendo la oferta recibida.
        """
        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== TRADE OFFER FROM ANOTHER PLAYER ===
{offer_desc}

Accept if: you RECEIVE more value than you GIVE (based on your current needs).
Reject if: it helps the offering player more than you, or you need what you'd give.

Reply with ONLY this JSON:
{{"accept": true}} or {{"accept": false}}"""

    @staticmethod
    def move_thief(game_state: str, current_thief_terrain: int, enemy_terrain_options: list) -> str:
        """
        Prompt para mover el ladrón.
        enemy_terrain_options: [(terrain_id, resource, dice, weight, enemy_player_id), ...]
        ordenadas por peso desc (mejores terrenos enemigos primero)
        """
        if enemy_terrain_options:
            options_text = "\n".join(
                f"  terrain {t_id}: {res} dice={dice} ({weight}/5 prob) — enemy P{pid}"
                for (t_id, res, dice, weight, pid) in enemy_terrain_options[:6]
            )
            options_section = f"Best enemy terrains to block (sorted by production value):\n{options_text}"
        else:
            options_section = "No enemy terrains visible — move to any high-value terrain."

        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== MOVE THE THIEF ===
Current thief at terrain: {current_thief_terrain}
Move to a DIFFERENT terrain to block the most productive enemy position.

{options_section}

Priority: terrain with dice 6 or 8 (★★★★★) where an enemy has a settlement/city, but NOT where you have one.

Reply with ONLY this JSON:
{{"terrain": <terrain_id 0-18>, "player": <player_id to steal from, or -1 if none>}}"""

    @staticmethod
    def discard_resources(game_state: str, hand_desc: str, must_discard: int, missing_goal: str) -> str:
        """Prompt para descartar cuando se tiene más de 7."""
        goal_text = f"Your build goal: {missing_goal}" if missing_goal else "No immediate goal."
        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== DISCARD RESOURCES (THIEF CALLED) ===
You have more than 7 resources. You MUST discard exactly {must_discard} cards.
Current hand: {hand_desc}

{goal_text}

Discard resources you have most of and need least for your goal.
Keep resources needed for your next build (city needs 2Wheat+3Ore, town needs 1each of Wh/Cl/W/Wo).

Reply with ONLY this JSON (amounts to DISCARD, must sum to {must_discard}):
{{"wheat": <int>, "ore": <int>, "clay": <int>, "wood": <int>, "wool": <int>}}"""

    @staticmethod
    def monopoly_card(game_state: str, opponent_totals: dict) -> str:
        """
        Prompt para carta de monopolio.
        opponent_totals: estimación de recursos que tienen los rivales (si disponible).
        """
        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== MONOPOLY CARD ===
Choose one resource type. ALL other players give you ALL their cards of that type.
0=Wheat, 1=Ore, 2=Clay, 3=Wood, 4=Wool

Choose the resource opponents likely have most of, OR the resource you need most.
Tip: common resources (Wheat, Ore) are usually abundant in late game.

Reply with ONLY this JSON:
{{"material": <0-4>}}"""

    @staticmethod
    def road_building_card(game_state: str, valid_roads: list) -> str:
        roads = [f"({r['starting_node']}→{r['finishing_node']})" for r in valid_roads[:8]]
        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== ROAD BUILDING CARD ===
Build 2 free roads. Valid positions (sorted toward high-value nodes): {roads}
Priority: build toward high-score nodes, harbors, or to expand your network.

Reply with ONLY this JSON:
{{"node_id": <int>, "road_to": <int>, "node_id_2": <int or null>, "road_to_2": <int or null>}}"""

    @staticmethod
    def year_of_plenty_card(game_state: str, missing_goal: str) -> str:
        goal_text = f"Your build goal: {missing_goal}" if missing_goal else "No immediate goal — pick most useful resources."
        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== YEAR OF PLENTY CARD ===
Take 2 free resources from the bank (can be the same type).
0=Wheat, 1=Ore, 2=Clay, 3=Wood, 4=Wool

{goal_text}
Pick the 2 resources that get you closest to your build goal.

Reply with ONLY this JSON:
{{"material": <0-4>, "material_2": <0-4>}}"""

    @staticmethod
    def parse_json_response(response: str) -> dict:
        """
        Extrae JSON de la respuesta del LLM.
        Intenta varios estrategias de parsing.
        """
        text = response.strip()

        # Intento 1: directo
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Intento 2: buscar primer { ... }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Intento 3: limpiar markdown
        import re
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned)
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            pass

        return None
