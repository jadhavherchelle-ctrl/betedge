from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .engine import BettingEngine
from .services import OddsAPIClient, StatsService
from .sports.baseball import load_baseball_data
from .sports.basketball import load_basketball_data
from .sports.tennis import load_tennis_data, normalize_tennis_player_name


@dataclass
class SportsSnapshot:
    baseball: dict
    basketball: dict
    tennis: dict


class BettingController:
    def __init__(self, stats_service: StatsService | None = None, engine: BettingEngine | None = None) -> None:
        self.stats_service = stats_service or StatsService()
        self.engine = engine or BettingEngine()

    def load_sports(self, client: OddsAPIClient, season: int, home_field_adv: float) -> SportsSnapshot:
        baseball = load_baseball_data(client, self.stats_service, self.engine, season, home_field_adv)
        basketball = load_basketball_data(client, self.stats_service, self.engine, home_field_adv)
        tennis = load_tennis_data(client, self.engine, home_field_adv)
        return SportsSnapshot(baseball=baseball, basketball=basketball, tennis=tennis)

    def normalize_tennis_name(self, name: str | None) -> str:
        return normalize_tennis_player_name(name)
