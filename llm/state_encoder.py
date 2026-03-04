"""
StateEncoder - Convierte el estado del juego Catan a texto natural
para que el LLM pueda entenderlo y razonar sobre él.
"""
import sys
sys.path.insert(0, '/root/.openclaw/workspace/PyCatan')

from Classes.Constants import MaterialConstants, BuildConstants, BuildMaterialsConstants


MATERIAL_NAMES = {
    MaterialConstants.CEREAL: 'Wheat',
    MaterialConstants.MINERAL: 'Ore',
    MaterialConstants.CLAY: 'Clay',
    MaterialConstants.WOOD: 'Wood',
    MaterialConstants.WOOL: 'Wool',
}

TERRAIN_NAMES = {
    0: 'Wheat field',
    1: 'Mountain (ore)',
    2: 'Hills (clay)',
    3: 'Forest (wood)',
    4: 'Pasture (wool)',
    -1: 'Desert',
}


class StateEncoder:
    """Convierte el estado del juego Catan a texto natural."""

    def encode_full_state(self, board, hand, dev_cards_hand, player_id: int, players: list) -> str:
        """Genera descripción completa del estado para el LLM."""
        lines = []
        lines.append("=== CATAN GAME STATE ===")
        lines.append(f"You are Player {player_id}.")
        lines.append("")

        # Recursos propios
        res = hand.resources
        total = res.cereal + res.mineral + res.clay + res.wood + res.wool
        lines.append(f"Your resources: Wheat={res.cereal}, Ore={res.mineral}, Clay={res.clay}, Wood={res.wood}, Wool={res.wool} (Total: {total})")

        # Puntos de victoria propios
        my_player = next((p for p in players if p['id'] == player_id), None)
        if my_player:
            vp = my_player.get('victory_points', 0)
            lines.append(f"Your victory points: {vp} / 10 (10 = WIN)")

        # Cartas de desarrollo
        if len(dev_cards_hand.hand) > 0:
            lines.append(f"Your development cards: {len(dev_cards_hand.hand)} cards in hand")

        # Otros jugadores
        others = [p for p in players if p['id'] != player_id]
        if others:
            other_desc = []
            for p in others:
                pid = p['id']
                pvp = p.get('victory_points', 0)
                other_desc.append(f"P{pid}({pvp} VP)")
            lines.append(f"Other players: {', '.join(other_desc)}")

        # Posiciones propias en el tablero
        my_nodes = [n['id'] for n in board.nodes if n.get('player') == player_id]
        if my_nodes:
            lines.append(f"Your settlements/cities at nodes: {my_nodes}")

        # Carreteras propias (roads: [{"player_id": int, "node_id": int}])
        my_roads = []
        seen = set()
        for node in board.nodes:
            for road in node.get('roads', []):
                if road['player_id'] == player_id:
                    edge = tuple(sorted([node['id'], road['node_id']]))
                    if edge not in seen:
                        seen.add(edge)
                        my_roads.append(f"{edge[0]}-{edge[1]}")
        if my_roads:
            lines.append(f"Your roads: {my_roads[:10]}{'...' if len(my_roads) > 10 else ''}")

        lines.append("")
        lines.append("=== WHAT YOU CAN BUILD ===")
        can_build = []
        if res.cereal >= 1 and res.mineral >= 1 and res.clay >= 1 and res.wood >= 1 and res.wool >= 1:
            can_build.append("town (1W+1C+1Cl+1Wh+1Wo)")
        if res.cereal >= 2 and res.mineral >= 3:
            can_build.append("city (2Wh+3Ore)")
        if res.clay >= 1 and res.wood >= 1:
            can_build.append("road (1W+1Cl)")
        if res.cereal >= 1 and res.mineral >= 1 and res.wool >= 1:
            can_build.append("development card (1Wh+1Ore+1Wo)")
        if can_build:
            lines.append(f"Affordable: {', '.join(can_build)}")
        else:
            lines.append("Cannot afford anything (need more resources)")

        return "\n".join(lines)

    def encode_hand(self, hand) -> str:
        """Descripción corta de la mano."""
        res = hand.resources
        return (f"Wheat={res.cereal}, Ore={res.mineral}, Clay={res.clay}, "
                f"Wood={res.wood}, Wool={res.wool}")

    def encode_valid_build_actions(self, board, player_id: int, hand) -> dict:
        """Retorna las acciones de construcción válidas."""
        res = hand.resources
        actions = {
            'can_build_town': False,
            'valid_town_nodes': [],
            'can_build_city': False,
            'valid_city_nodes': [],
            'can_build_road': False,
            'valid_road_nodes': [],
            'can_buy_card': False,
        }

        # Town
        can_afford_town = res.cereal >= 1 and res.mineral >= 0 and res.clay >= 1 and res.wood >= 1 and res.wool >= 1
        if can_afford_town:
            valid = board.valid_town_nodes(player_id)
            if valid:
                actions['can_build_town'] = True
                actions['valid_town_nodes'] = valid[:10]

        # City
        can_afford_city = res.cereal >= 2 and res.mineral >= 3
        if can_afford_city:
            valid = board.valid_city_nodes(player_id)
            if valid:
                actions['can_build_city'] = True
                actions['valid_city_nodes'] = valid[:10]

        # Road
        can_afford_road = res.clay >= 1 and res.wood >= 1
        if can_afford_road:
            valid = board.valid_road_nodes(player_id)
            if valid:
                actions['can_build_road'] = True
                actions['valid_road_nodes'] = valid[:5]

        # Card
        if res.cereal >= 1 and res.mineral >= 1 and res.wool >= 1:
            actions['can_buy_card'] = True

        return actions
