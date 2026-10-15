from __future__ import annotations

import pandas as pd

from ..engine import BettingEngine
from ..services import OddsAPIClient


def normalize_tennis_player_name(name: str | None) -> str:
    return "" if name is None else str(name).strip()


def build_tennis_stats_from_odds(odds_games: list[dict]) -> pd.DataFrame:
    players: set[str] = set()
    for game in odds_games:
        home = normalize_tennis_player_name(game.get("home_team"))
        away = normalize_tennis_player_name(game.get("away_team"))
        if home:
            players.add(home)
        if away:
            players.add(away)

    return pd.DataFrame([{ "team": player, "power_score": 0.0 } for player in sorted(players)])


def load_tennis_data(
    client: OddsAPIClient,
    engine: BettingEngine,
    home_field_adv: float,
) -> dict:
    odds = client.fetch_active_tennis_odds(
        regions_priority=("us", "uk", "eu", "au"),
        include_sport_keys=("tennis_atp_madrid_open", "tennis_wta_madrid_open"),
    )
    stats = build_tennis_stats_from_odds(odds)
    edges = engine.build_game_edges(
        odds,
        stats,
        home_field_adv,
        normalize_tennis_player_name,
    )
    return {"odds": odds, "stats": stats, "edges": edges}
