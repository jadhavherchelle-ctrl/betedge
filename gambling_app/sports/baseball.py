from __future__ import annotations

from ..constants import MLB_ODDS_SPORT
from ..engine import BettingEngine
from ..services import OddsAPIClient, StatsService


def load_baseball_data(
    client: OddsAPIClient,
    stats_service: StatsService,
    engine: BettingEngine,
    season: int,
    home_field_adv: float,
) -> dict:
    odds = client.fetch_live_odds(MLB_ODDS_SPORT)
    stats = stats_service.fetch_mlb_team_stats(int(season))
    context = stats_service.fetch_mlb_game_context(odds, int(season))
    edges = engine.build_game_edges(
        odds,
        stats,
        home_field_adv,
        BettingEngine.normalize_mlb_team_name,
        game_context=context,
    )
    return {"odds": odds, "stats": stats, "context": context, "edges": edges}
