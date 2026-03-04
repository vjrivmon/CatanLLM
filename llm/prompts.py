"""
PromptBuilder - Plantillas de prompts para cada fase del juego Catan.
El LLM debe responder SIEMPRE con JSON válido.
"""
import json


SYSTEM_PREAMBLE = """You are an expert Catan player. You make strategic decisions to win.
Goal: reach 10 victory points first.
You MUST reply with ONLY valid JSON, no explanation, no markdown, just JSON."""


class PromptBuilder:

    @staticmethod
    def build_phase(game_state: str, actions: dict) -> str:
        valid_actions = []
        if actions['can_build_town']:
            valid_actions.append(f"town at one of nodes: {actions['valid_town_nodes']}")
        if actions['can_build_city']:
            valid_actions.append(f"city at one of nodes: {actions['valid_city_nodes']}")
        if actions['can_build_road']:
            roads = [f"({r['starting_node']}->{r['finishing_node']})" for r in actions['valid_road_nodes']]
            valid_actions.append(f"road: {roads}")
        if actions['can_buy_card']:
            valid_actions.append("development card")
        if not valid_actions:
            valid_actions.append("none (cannot afford anything)")

        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== BUILD PHASE ===
Valid actions: {'; '.join(valid_actions)}

Choose the best strategic action. Reply with ONLY this JSON:
{{"action": "town|city|road|card|none", "node_id": <integer or null>, "road_to": <integer or null>}}

Examples:
{{"action": "town", "node_id": 15, "road_to": null}}
{{"action": "road", "node_id": 5, "road_to": 6}}
{{"action": "none", "node_id": null, "road_to": null}}"""

    @staticmethod
    def game_start(board_info: str, valid_start_nodes: list) -> str:
        nodes_sample = valid_start_nodes[:15] if len(valid_start_nodes) > 15 else valid_start_nodes
        return f"""{SYSTEM_PREAMBLE}

=== GAME START - PLACE YOUR FIRST SETTLEMENT ===
{board_info}
Available nodes to place your first settlement: {nodes_sample}

Pick the node with best access to diverse resources (wheat, ore, wood, clay, wool).
Also pick an adjacent road direction.

Reply with ONLY this JSON:
{{"node_id": <integer>, "road_to": <integer>}}"""

    @staticmethod
    def commerce_phase(game_state: str) -> str:
        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== COMMERCE PHASE ===
You can trade 4:1 with the bank (give 4 of same resource, receive 1 of any).
Materials: 0=Wheat, 1=Ore, 2=Clay, 3=Wood, 4=Wool

Should you trade? Only trade if it helps you build something.

Reply with ONLY this JSON (or null to skip):
{{"gives": <material_int 0-4>, "receives": <material_int 0-4>}}
Or: {{"action": "skip"}}"""

    @staticmethod
    def move_thief(game_state: str, current_thief_terrain: int) -> str:
        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== MOVE THE THIEF ===
Current thief location: terrain {current_thief_terrain}
Move to a different terrain to block the most dangerous opponent.
Terrain IDs: 0-18 (avoid terrains where YOU have settlements).

Reply with ONLY this JSON:
{{"terrain": <terrain_id 0-18>, "player": <player_id to steal from, or -1 if none>}}"""

    @staticmethod
    def discard_resources(game_state: str, hand_desc: str, must_discard: int) -> str:
        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== DISCARD RESOURCES ===
You have more than 7 resources. You must discard {must_discard} cards.
Current hand: {hand_desc}
Keep resources that help you build toward your goal.

Reply with ONLY this JSON (amounts to DISCARD):
{{"wheat": <int>, "ore": <int>, "clay": <int>, "wood": <int>, "wool": <int>}}"""

    @staticmethod
    def monopoly_card(game_state: str) -> str:
        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== MONOPOLY CARD ===
Choose one resource type. ALL other players give you ALL their cards of that type.
0=Wheat, 1=Ore, 2=Clay, 3=Wood, 4=Wool

Choose the resource you think others have most of, or the one you need most.
Reply with ONLY this JSON:
{{"material": <0-4>}}"""

    @staticmethod
    def road_building_card(game_state: str, valid_roads: list) -> str:
        roads = [f"({r['starting_node']}->{r['finishing_node']})" for r in valid_roads[:8]]
        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== ROAD BUILDING CARD ===
Build 2 free roads. Valid positions: {roads}
Build toward unexplored areas or to expand your network.

Reply with ONLY this JSON:
{{"node_id": <int>, "road_to": <int>, "node_id_2": <int or null>, "road_to_2": <int or null>}}"""

    @staticmethod
    def year_of_plenty_card(game_state: str) -> str:
        return f"""{SYSTEM_PREAMBLE}

{game_state}

=== YEAR OF PLENTY CARD ===
Take 2 free resources from the bank (can be the same).
0=Wheat, 1=Ore, 2=Clay, 3=Wood, 4=Wool

Choose the 2 resources you need most to build your next item.
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
