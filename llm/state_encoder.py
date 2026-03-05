"""
StateEncoder - Convierte el estado del juego Catan a texto natural
para que el LLM pueda entenderlo y razonar sobre él.

v2: Incluye terrenos + probabilidades + recursos por nodo + puertos + posiciones rivales.
"""
import sys
sys.path.insert(0, '/root/.openclaw/workspace/PyCatan')

from Classes.Constants import MaterialConstants, BuildConstants, BuildMaterialsConstants, TerrainConstants, HarborConstants


MATERIAL_NAMES = {
    MaterialConstants.CEREAL: 'Wheat',
    MaterialConstants.MINERAL: 'Ore',
    MaterialConstants.CLAY:   'Clay',
    MaterialConstants.WOOD:   'Wood',
    MaterialConstants.WOOL:   'Wool',
}

TERRAIN_RESOURCE = {
    TerrainConstants.CEREAL:  'Wheat',
    TerrainConstants.MINERAL: 'Ore',
    TerrainConstants.CLAY:    'Clay',
    TerrainConstants.WOOD:    'Wood',
    TerrainConstants.WOOL:    'Wool',
    TerrainConstants.DESERT:  'Desert(no prod)',
}

HARBOR_NAMES = {
    HarborConstants.CEREAL:  'Wheat 2:1',
    HarborConstants.MINERAL: 'Ore 2:1',
    HarborConstants.CLAY:    'Clay 2:1',
    HarborConstants.WOOD:    'Wood 2:1',
    HarborConstants.WOOL:    'Wool 2:1',
    HarborConstants.ALL:     'Any 3:1',
    HarborConstants.NONE:    None,
}

# Pesos estadísticos: nº de combinaciones de dados que generan ese número
DICE_WEIGHT = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


class StateEncoder:
    """Convierte el estado del juego Catan a texto natural enriquecido."""

    # ── Helpers internos ─────────────────────────────────────────────────────

    def _node_production(self, board, node_id: int) -> list:
        """
        Retorna lista de (resource_name, dice_number, stat_weight) para cada terreno
        que toca el nodo. Excluye desierto.
        """
        result = []
        for t_id in board.nodes[node_id]['contacting_terrain']:
            t = board.terrain[t_id]
            if t['terrain_type'] == TerrainConstants.DESERT:
                continue
            rname = TERRAIN_RESOURCE.get(t['terrain_type'], '?')
            dice  = t['probability']
            weight = DICE_WEIGHT.get(dice, 0)
            result.append((rname, dice, weight))
        return result

    def _node_score(self, board, node_id: int) -> int:
        """Puntuación estadística del nodo: suma de pesos de dados de sus terrenos."""
        return sum(w for _, _, w in self._node_production(board, node_id))

    def _node_summary(self, board, node_id: int) -> str:
        """Texto compacto: 'node 12 → Wheat(6★★★★★) Ore(9★★★★)'"""
        prods = self._node_production(board, node_id)
        if not prods:
            return f"node {node_id} → (no production)"
        parts = []
        for rname, dice, weight in prods:
            stars = '★' * weight
            parts.append(f"{rname}({dice}{stars})")
        score = sum(w for _, _, w in prods)
        port = board.nodes[node_id].get('harbor', HarborConstants.NONE)
        port_str = f" [PORT:{HARBOR_NAMES[port]}]" if port != HarborConstants.NONE and HARBOR_NAMES.get(port) else ""
        return f"node {node_id} (score={score}) → {' + '.join(parts)}{port_str}"

    def _player_ports(self, board, player_id: int) -> list:
        """Lista de puertos que tiene el jugador."""
        ports = []
        for node in board.nodes:
            if node['player'] == player_id:
                h = node.get('harbor', HarborConstants.NONE)
                name = HARBOR_NAMES.get(h)
                if name:
                    ports.append(name)
        return list(set(ports))

    def _trade_ratio(self, board, player_id: int, material: int) -> int:
        """Ratio de comercio óptimo para un material (2, 3 o 4)."""
        # Puerto específico 2:1
        if board.check_for_player_harbors(player_id, material) == material:
            return 2
        # Puerto 3:1
        if board.check_for_player_harbors(player_id, HarborConstants.ALL) == HarborConstants.ALL:
            return 3
        return 4

    # ── API pública ───────────────────────────────────────────────────────────

    def encode_full_state(self, board, hand, dev_cards_hand, player_id: int, players: list) -> str:
        """Genera descripción completa del estado para el LLM (v2)."""
        lines = []
        lines.append("=== CATAN GAME STATE ===")
        lines.append(f"You are Player {player_id}.")
        lines.append("")

        # Recursos propios
        res = hand.resources
        total = res.cereal + res.mineral + res.clay + res.wood + res.wool
        lines.append(f"Your hand: Wheat={res.cereal}, Ore={res.mineral}, Clay={res.clay}, Wood={res.wood}, Wool={res.wool} (Total: {total})")

        # VP y objetivo
        my_player = next((p for p in players if p['id'] == player_id), None)
        if my_player:
            vp = my_player.get('victory_points', 0)
            lines.append(f"Your VP: {vp}/10 (need 10 to WIN)")

        # Cartas de desarrollo
        dev_hand = dev_cards_hand.hand if dev_cards_hand else []
        if dev_hand:
            from Classes.Constants import DevelopmentCardConstants as DCC
            knights    = sum(1 for c in dev_hand if c.type == DCC.KNIGHT)
            vp_cards   = sum(1 for c in dev_hand if c.type == DCC.VICTORY_POINT)
            progress   = len(dev_hand) - knights - vp_cards
            parts = []
            if knights:  parts.append(f"{knights}xKnight")
            if vp_cards: parts.append(f"{vp_cards}xVP")
            if progress: parts.append(f"{progress}xProgress")
            lines.append(f"Dev cards in hand: {', '.join(parts)}")

        # Puertos propios
        ports = self._player_ports(board, player_id)
        if ports:
            lines.append(f"Your ports: {', '.join(ports)}")
        else:
            lines.append("Your ports: none (all trades at 4:1)")

        lines.append("")

        # Posiciones propias con detalle de producción
        my_nodes = [(n['id'], n.get('has_city', False))
                    for n in board.nodes if n.get('player') == player_id]
        if my_nodes:
            lines.append("Your settlements/cities:")
            for (nid, is_city) in my_nodes:
                btype = "CITY" if is_city else "town"
                lines.append(f"  {btype}: {self._node_summary(board, nid)}")
        else:
            lines.append("Your settlements/cities: none yet")

        # Rivales
        others = [p for p in players if p['id'] != player_id]
        if others:
            lines.append("")
            lines.append("Opponents:")
            for p in others:
                pid = p['id']
                pvp = p.get('victory_points', 0)
                p_nodes = [(n['id'], n.get('has_city', False))
                           for n in board.nodes if n.get('player') == pid]
                node_strs = []
                for (nid, is_city) in p_nodes:
                    btype = "city" if is_city else "town"
                    score = self._node_score(board, nid)
                    node_strs.append(f"node{nid}(score={score}{'★' if is_city else ''})")
                p_nodes_str = ', '.join(node_strs) if node_strs else 'none'
                lines.append(f"  P{pid}: {pvp} VP | positions: {p_nodes_str}")

        # Ladrón
        thief_terrain = next((t for t in board.terrain if t.get('has_thief')), None)
        if thief_terrain:
            rname = TERRAIN_RESOURCE.get(thief_terrain['terrain_type'], '?')
            lines.append(f"\nThief at terrain {thief_terrain['id']} ({rname}, dice={thief_terrain['probability']})")

        lines.append("")
        lines.append("=== WHAT YOU CAN BUILD ===")
        can_build = []
        if res.cereal >= 1 and res.clay >= 1 and res.wood >= 1 and res.wool >= 1:
            can_build.append("town (1Wh+1Cl+1W+1Wo)")
        if res.cereal >= 2 and res.mineral >= 3:
            can_build.append("city (2Wh+3Ore)")
        if res.clay >= 1 and res.wood >= 1:
            can_build.append("road (1W+1Cl)")
        if res.cereal >= 1 and res.mineral >= 1 and res.wool >= 1:
            can_build.append("dev card (1Wh+1Ore+1Wo)")
        lines.append(f"Affordable: {', '.join(can_build) if can_build else 'nothing'}")

        # Qué le falta para siguiente construcción útil
        missing = self._what_is_missing(res)
        if missing:
            lines.append(f"Closest goal: {missing}")

        return "\n".join(lines)

    def _what_is_missing(self, res) -> str:
        """Texto de qué falta para la siguiente construcción más cercana."""
        # ¿Falta poco para ciudad?
        need_city_wh  = max(0, 2 - res.cereal)
        need_city_ore = max(0, 3 - res.mineral)
        if need_city_wh + need_city_ore <= 2 and (res.cereal > 0 or res.mineral > 0):
            parts = []
            if need_city_wh:  parts.append(f"{need_city_wh}Wheat")
            if need_city_ore: parts.append(f"{need_city_ore}Ore")
            return f"city (need: {'+'.join(parts)})" if parts else ""

        # ¿Falta poco para pueblo?
        need_town = (max(0, 1 - res.cereal) + max(0, 1 - res.clay) +
                     max(0, 1 - res.wood)  + max(0, 1 - res.wool))
        if need_town <= 2:
            parts = []
            if res.cereal < 1: parts.append("1Wheat")
            if res.clay   < 1: parts.append("1Clay")
            if res.wood   < 1: parts.append("1Wood")
            if res.wool   < 1: parts.append("1Wool")
            return f"town (need: {'+'.join(parts)})" if parts else ""

        return ""

    def encode_hand(self, hand) -> str:
        """Descripción corta de la mano."""
        res = hand.resources
        return (f"Wheat={res.cereal}, Ore={res.mineral}, Clay={res.clay}, "
                f"Wood={res.wood}, Wool={res.wool}")

    def encode_valid_build_actions(self, board, player_id: int, hand) -> dict:
        """
        Retorna las acciones de construcción válidas, enriquecidas con
        scores de terreno para cada nodo candidato.
        """
        res = hand.resources
        actions = {
            'can_build_town': False,
            'valid_town_nodes': [],
            'valid_town_nodes_scored': [],   # [(node_id, score, summary), ...]
            'can_build_city': False,
            'valid_city_nodes': [],
            'valid_city_nodes_scored': [],
            'can_build_road': False,
            'valid_road_nodes': [],
            'can_buy_card': False,
        }

        # Town
        if res.cereal >= 1 and res.clay >= 1 and res.wood >= 1 and res.wool >= 1:
            valid = board.valid_town_nodes(player_id)
            if valid:
                actions['can_build_town'] = True
                # Ordenar por score desc y limitar a 8
                scored = sorted(
                    [(nid, self._node_score(board, nid), self._node_summary(board, nid))
                     for nid in valid],
                    key=lambda x: -x[1]
                )[:8]
                actions['valid_town_nodes'] = [s[0] for s in scored]
                actions['valid_town_nodes_scored'] = scored

        # City
        if res.cereal >= 2 and res.mineral >= 3:
            valid = board.valid_city_nodes(player_id)
            if valid:
                actions['can_build_city'] = True
                scored = sorted(
                    [(nid, self._node_score(board, nid), self._node_summary(board, nid))
                     for nid in valid],
                    key=lambda x: -x[1]
                )[:8]
                actions['valid_city_nodes'] = [s[0] for s in scored]
                actions['valid_city_nodes_scored'] = scored

        # Road
        if res.clay >= 1 and res.wood >= 1:
            valid = board.valid_road_nodes(player_id)
            if valid:
                actions['can_build_road'] = True
                # Puntuar carretera por el score del nodo destino
                scored_roads = sorted(
                    valid,
                    key=lambda r: -self._node_score(board, r['finishing_node'])
                )[:6]
                actions['valid_road_nodes'] = scored_roads

        # Card
        if res.cereal >= 1 and res.mineral >= 1 and res.wool >= 1:
            actions['can_buy_card'] = True

        return actions

    def encode_trade_ratios(self, board, player_id: int) -> dict:
        """Retorna el ratio 4:1, 3:1 o 2:1 para cada material."""
        return {
            'Wheat': self._trade_ratio(board, player_id, MaterialConstants.CEREAL),
            'Ore':   self._trade_ratio(board, player_id, MaterialConstants.MINERAL),
            'Clay':  self._trade_ratio(board, player_id, MaterialConstants.CLAY),
            'Wood':  self._trade_ratio(board, player_id, MaterialConstants.WOOD),
            'Wool':  self._trade_ratio(board, player_id, MaterialConstants.WOOL),
        }
