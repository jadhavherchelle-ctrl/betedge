from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from gambling_app.constants import MLB_ODDS_SPORT, ODDS_API_BASE
from gambling_app.engine import BettingEngine
from gambling_app.services import StatsService


def fetch_historical_mlb_odds(api_key: str, snapshot_iso_utc: str) -> list[dict]:
    url = f"{ODDS_API_BASE}/historical/sports/{MLB_ODDS_SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "date": snapshot_iso_utc,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("odds"), list):
            return payload["odds"]
    if isinstance(payload, list):
        return payload
    return []


def fetch_mlb_results_for_date(target_date: date) -> dict[str, str]:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": target_date.isoformat(),
        "hydrate": "linescore,team",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    winners: dict[str, str] = {}
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            status = game.get("status", {}).get("codedGameState")
            if status not in {"F", "O"}:
                continue

            home_raw = game.get("teams", {}).get("home", {}).get("team", {}).get("name")
            away_raw = game.get("teams", {}).get("away", {}).get("team", {}).get("name")
            home = BettingEngine.normalize_mlb_team_name(home_raw)
            away = BettingEngine.normalize_mlb_team_name(away_raw)
            key = f"{away} @ {home}"

            home_runs = game.get("teams", {}).get("home", {}).get("score")
            away_runs = game.get("teams", {}).get("away", {}).get("score")
            if home_runs is None or away_runs is None:
                continue

            if int(home_runs) > int(away_runs):
                winners[key] = home
            elif int(away_runs) > int(home_runs):
                winners[key] = away

    return winners


def simulate_yesterday(min_edge_pct: float, home_field_adv: float, season: int | None) -> int:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        print("Missing ODDS_API_KEY environment variable.")
        return 2

    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    snapshot_dt = datetime(yesterday.year, yesterday.month, yesterday.day, 17, 0, 0, tzinfo=timezone.utc)
    snapshot_iso_utc = snapshot_dt.isoformat().replace("+00:00", "Z")

    active_season = int(season or yesterday.year)

    print(f"Loading historical odds snapshot: {snapshot_iso_utc}")
    odds_games = fetch_historical_mlb_odds(api_key=api_key, snapshot_iso_utc=snapshot_iso_utc)
    if not odds_games:
        print("No historical MLB odds returned for that snapshot.")
        return 1

    stats_service = StatsService()
    stats_df = stats_service.fetch_mlb_team_stats(active_season)
    game_context = stats_service.fetch_mlb_game_context(odds_games, active_season)

    edges = BettingEngine.build_game_edges(
        odds_games=odds_games,
        stats_df=stats_df,
        home_field_adv=float(home_field_adv),
        normalizer=BettingEngine.normalize_mlb_team_name,
        game_context=game_context,
    )

    if edges.empty:
        print("No edge rows were generated.")
        return 1

    picks = edges[edges["edge"] >= (float(min_edge_pct) / 100)].copy()
    if picks.empty:
        print(f"No picks found at >= {min_edge_pct:.2f}% edge.")
        return 0

    winners_by_game = fetch_mlb_results_for_date(yesterday)
    if not winners_by_game:
        print("Could not load final MLB results for yesterday.")
        return 1

    picks["winner"] = picks["game"].map(winners_by_game)
    picks = picks[picks["winner"].notna()].copy()
    if picks.empty:
        print("No picks could be matched to final game results.")
        return 1

    picks["won"] = picks["team"] == picks["winner"]
    picks["profit_u"] = picks.apply(
        lambda row: (float(row["decimal_odds"]) - 1.0) if bool(row["won"]) else -1.0,
        axis=1,
    )

    total_bets = len(picks)
    wins = int(picks["won"].sum())
    losses = total_bets - wins
    total_profit_u = float(picks["profit_u"].sum())
    roi_pct = (total_profit_u / total_bets) * 100 if total_bets else 0.0

    out = picks[["game", "team", "american_odds", "edge", "winner", "won", "profit_u"]].copy()
    out["edge"] = (out["edge"] * 100).round(2)
    out["profit_u"] = out["profit_u"].round(3)
    print("\n=== Yesterday Replay (MLB) ===")
    print(out.sort_values(by="edge", ascending=False).to_string(index=False))
    print("\n=== Summary ===")
    print(f"Date: {yesterday.isoformat()}")
    print(f"Snapshot: {snapshot_iso_utc}")
    print(f"Min edge: {min_edge_pct:.2f}%")
    print(f"Bets: {total_bets} | Wins: {wins} | Losses: {losses}")
    print(f"Profit: {total_profit_u:+.3f}u")
    print(f"ROI: {roi_pct:+.2f}%")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay yesterday's MLB picks using historical odds and final scores.")
    parser.add_argument("--min-edge", type=float, default=2.0, help="Minimum edge percent for qualifying picks.")
    parser.add_argument("--home-field-adv", type=float, default=0.06, help="Home-field model adjustment.")
    parser.add_argument("--season", type=int, default=None, help="Season year override. Default: yesterday's year.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(simulate_yesterday(args.min_edge, args.home_field_adv, args.season))
