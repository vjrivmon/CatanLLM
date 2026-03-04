"""
GameMetrics - Almacena y reporta métricas de una partida o benchmark.
"""
import time
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnMetric:
    turn_number: int
    player_id: int
    phase: str
    decision_time_s: float
    action_taken: str
    was_llm: bool
    was_fallback: bool


@dataclass
class GameMetrics:
    game_id: int
    llm_player_id: int
    model_name: str
    winner_id: Optional[int] = None
    total_turns: int = 0
    total_time_s: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    llm_decisions: int = 0
    fallback_decisions: int = 0
    decision_times: list = field(default_factory=list)

    def finish(self, winner_id: int):
        self.winner_id = winner_id
        self.end_time = time.time()
        self.total_time_s = round(self.end_time - self.start_time, 2)

    @property
    def llm_won(self) -> bool:
        return self.winner_id == self.llm_player_id

    @property
    def avg_decision_time(self) -> float:
        if not self.decision_times:
            return 0.0
        return round(sum(self.decision_times) / len(self.decision_times), 3)

    @property
    def max_decision_time(self) -> float:
        return round(max(self.decision_times), 3) if self.decision_times else 0.0

    @property
    def fallback_rate(self) -> float:
        total = self.llm_decisions + self.fallback_decisions
        if total == 0:
            return 0.0
        return round(self.fallback_decisions / total, 3)

    def to_dict(self) -> dict:
        return {
            'game_id': self.game_id,
            'llm_player_id': self.llm_player_id,
            'model_name': self.model_name,
            'winner_id': self.winner_id,
            'llm_won': self.llm_won,
            'total_turns': self.total_turns,
            'total_time_s': self.total_time_s,
            'llm_decisions': self.llm_decisions,
            'fallback_decisions': self.fallback_decisions,
            'fallback_rate': self.fallback_rate,
            'avg_decision_time_s': self.avg_decision_time,
            'max_decision_time_s': self.max_decision_time,
        }

    def print_summary(self):
        print("\n" + "=" * 50)
        print(f"  GAME {self.game_id} SUMMARY")
        print("=" * 50)
        print(f"  Model:          {self.model_name}")
        print(f"  LLM Player:     P{self.llm_player_id}")
        print(f"  Winner:         P{self.winner_id} {'🏆 LLM WON!' if self.llm_won else '(Random won)'}")
        print(f"  Total turns:    {self.total_turns}")
        print(f"  Total time:     {self.total_time_s}s")
        print(f"  LLM decisions:  {self.llm_decisions}")
        print(f"  Fallbacks:      {self.fallback_decisions} ({self.fallback_rate*100:.1f}%)")
        print(f"  Avg LLM time:   {self.avg_decision_time}s/decision")
        print(f"  Max LLM time:   {self.max_decision_time}s")
        print("=" * 50)


class BenchmarkSummary:
    def __init__(self, games: list[GameMetrics]):
        self.games = games

    @property
    def total_games(self) -> int:
        return len(self.games)

    @property
    def llm_wins(self) -> int:
        return sum(1 for g in self.games if g.llm_won)

    @property
    def win_rate(self) -> float:
        if not self.games:
            return 0.0
        return round(self.llm_wins / self.total_games, 3)

    @property
    def avg_game_time(self) -> float:
        if not self.games:
            return 0.0
        return round(sum(g.total_time_s for g in self.games) / len(self.games), 2)

    @property
    def avg_decision_time(self) -> float:
        all_times = []
        for g in self.games:
            all_times.extend(g.decision_times)
        return round(sum(all_times) / len(all_times), 3) if all_times else 0.0

    def print_summary(self):
        print("\n" + "=" * 60)
        print("  BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"  Total games:        {self.total_games}")
        print(f"  LLM wins:           {self.llm_wins} / {self.total_games} ({self.win_rate*100:.1f}%)")
        print(f"  Avg game time:      {self.avg_game_time}s")
        print(f"  Avg decision time:  {self.avg_decision_time}s/decision")
        print("=" * 60)
