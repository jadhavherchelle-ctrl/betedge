from __future__ import annotations

from ..constants import NBA_ODDS_SPORT
from ..engine import BettingEngine
from ..services import OddsAPIClient, StatsService


def load_basketball_data(
    client: OddsAPIClient,
    stats_service: StatsService,
    engine: BettingEngine,
    home_field_adv: float,
) -> dict:
    odds = client.fetch_live_odds(NBA_ODDS_SPORT)
    stats = stats_service.fetch_nba_team_stats()
    edges = engine.build_game_edges(
        odds,
        stats,
        home_field_adv,
        BettingEngine.normalize_nba_team_name,
    )
    return {"odds": odds, "stats": stats, "edges": edges}
