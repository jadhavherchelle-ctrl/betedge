"""Multi-day MLB backtest engine.

Fetches historical odds snapshots, runs BettingEngine edge scoring, grades picks
against actual results, and accumulates per-day and summary statistics.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from .constants import MLB_ODDS_SPORT, ODDS_API_BASE
from .engine import BettingEngine
from .services import StatsService


# ── API helpers ───────────────────────────────────────────────────────────────

class HistoricalOddsUnavailable(Exception):
    """Raised when the Odds API rejects the historical endpoint (e.g. free-tier key)."""


def _fetch_historical_odds(api_key: str, snapshot_iso_utc: str) -> list[dict]:
    url = f"{ODDS_API_BASE}/historical/sports/{MLB_ODDS_SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "date": snapshot_iso_utc,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 401:
        raise HistoricalOddsUnavailable(
            "The Odds API historical endpoint requires a paid subscription. "
            "Free-tier keys only support live odds."
        )
    if resp.status_code == 422:
        raise HistoricalOddsUnavailable(
            "Historical odds not available for this date (too far back or future)."
        )
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("odds"), list):
            return payload["odds"]
    if isinstance(payload, list):
        return payload
    return []


def _fetch_mlb_results(target_date: date) -> dict[str, str]:
    """Return {game_key: winner_name} for all finished MLB games on target_date."""
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": target_date.isoformat(), "hydrate": "linescore,team"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    winners: dict[str, str] = {}
    for day in resp.json().get("dates", []):
        for game in day.get("games", []):
            if game.get("status", {}).get("codedGameState") not in {"F", "O"}:
                continue
            home_raw = game["teams"]["home"]["team"]["name"]
            away_raw = game["teams"]["away"]["team"]["name"]
            home = BettingEngine.normalize_mlb_team_name(home_raw)
            away = BettingEngine.normalize_mlb_team_name(away_raw)
            h_runs = game["teams"]["home"].get("score")
            a_runs = game["teams"]["away"].get("score")
            if h_runs is None or a_runs is None:
                continue
            key = f"{away} @ {home}"
            winners[key] = home if int(h_runs) > int(a_runs) else away
    return winners


# ── Single-day simulation ─────────────────────────────────────────────────────

def simulate_day(
    target_date: date,
    api_key: str,
    stats_df: pd.DataFrame,
    min_edge_pct: float = 2.0,
    home_field_adv: float = 0.06,
    build_parlays: bool = False,
    parlay_legs: int = 3,
    parlay_top_n: int = 8,
    bankroll: float = 10.0,
    max_parlays: int = 5,
    fractional_kelly: float = 0.35,
) -> dict:
    """
    Simulate a single date. stats_df should be pre-loaded to avoid repeated fetches.

    Returns a dict with keys:
      date, skipped, reason (if skipped), picks, wins, losses, win_rate_pct,
      profit_u, roi_pct, avg_edge_pct, avg_model_prob_pct, picks_df,
      and optionally parlay_count, parlay_wins, parlay_staked, parlay_profit,
      parlay_roi_pct, parlay_df.
    """
    snapshot_dt = datetime(
        target_date.year, target_date.month, target_date.day, 17, 0, 0, tzinfo=timezone.utc
    )
    snapshot_iso = snapshot_dt.isoformat().replace("+00:00", "Z")

    try:
        odds_games = _fetch_historical_odds(api_key=api_key, snapshot_iso_utc=snapshot_iso)
    except HistoricalOddsUnavailable as e:
        return {"date": target_date.isoformat(), "skipped": True, "reason": str(e), "auth_error": True}
    except Exception as e:
        return {"date": target_date.isoformat(), "skipped": True, "reason": f"Odds API: {e}"}

    if not odds_games:
        return {"date": target_date.isoformat(), "skipped": True, "reason": "No odds data returned"}

    edges = BettingEngine.build_game_edges(
        odds_games=odds_games,
        stats_df=stats_df,
        home_field_adv=home_field_adv,
        normalizer=BettingEngine.normalize_mlb_team_name,
        game_context={},
    )

    if edges.empty:
        return {"date": target_date.isoformat(), "skipped": True, "reason": "No edges generated"}

    picks = BettingEngine.filter_realistic(edges, min_edge_pct, 50.0)
    if picks.empty:
        return {
            "date": target_date.isoformat(),
            "skipped": True,
            "reason": f"No picks at ≥{min_edge_pct:.1f}% edge",
        }

    try:
        winners = _fetch_mlb_results(target_date)
    except Exception as e:
        return {"date": target_date.isoformat(), "skipped": True, "reason": f"Results API: {e}"}

    if not winners:
        return {"date": target_date.isoformat(), "skipped": True, "reason": "No completed games found"}

    picks["winner"] = picks["game"].map(winners)
    picks = picks[picks["winner"].notna()].copy()
    if picks.empty:
        return {
            "date": target_date.isoformat(),
            "skipped": True,
            "reason": "No picks matched finished games",
        }

    picks["won"] = picks["team"] == picks["winner"]
    picks["profit_u"] = picks.apply(
        lambda r: float(r["decimal_odds"]) - 1.0 if bool(r["won"]) else -1.0, axis=1
    )

    n = len(picks)
    wins = int(picks["won"].sum())
    profit = float(picks["profit_u"].sum())

    result: dict = {
        "date": target_date.isoformat(),
        "skipped": False,
        "reason": "",
        "picks": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate_pct": round(wins / n * 100, 1) if n else 0.0,
        "profit_u": round(profit, 3),
        "roi_pct": round(profit / n * 100, 2) if n else 0.0,
        "avg_edge_pct": round(float(picks["edge"].mean()) * 100, 2),
        "avg_model_prob_pct": round(float(picks["model_prob"].mean()) * 100, 1),
        "picks_df": picks[
            ["game", "team", "american_odds", "edge", "model_prob", "winner", "won", "profit_u"]
        ].copy(),
    }

    if build_parlays and n >= 2:
        try:
            parlay_df = BettingEngine.build_parlays(picks, max_legs=parlay_legs, top_n=parlay_top_n)
            if not parlay_df.empty:
                alloc = BettingEngine.allocate_bankroll_to_parlays(
                    parlay_df,
                    bankroll=bankroll,
                    max_parlays=max_parlays,
                    max_bet_pct=0.40,
                    fractional_kelly=fractional_kelly,
                    diversification_bias=0.85,
                    max_team_uses=2,
                    max_team_exposure_pct=0.40,
                    contrarian_mix_pct=0.20,
                )
                if not alloc.empty:
                    p_staked = p_profit = 0.0
                    p_wins = 0
                    p_rows = []
                    for _, row in alloc.iterrows():
                        teams = list(row["teams"])
                        games = list(row["games"])
                        won = all(winners.get(g) == t for g, t in zip(games, teams))
                        stake = float(row["stake"])
                        pnl = (stake * float(row["decimal_odds"]) - stake) if won else -stake
                        p_staked += stake
                        p_profit += pnl
                        if won:
                            p_wins += 1
                        p_rows.append(
                            {
                                "parlay": row["parlay"],
                                "legs": int(row.get("legs", 0)),
                                "stake": round(stake, 2),
                                "decimal_odds": round(float(row["decimal_odds"]), 2),
                                "model_prob_pct": round(float(row.get("model_prob", 0)) * 100, 1),
                                "won": won,
                                "profit": round(pnl, 3),
                            }
                        )
                    result["parlay_count"] = len(p_rows)
                    result["parlay_wins"] = p_wins
                    result["parlay_staked"] = round(p_staked, 2)
                    result["parlay_profit"] = round(p_profit, 3)
                    result["parlay_roi_pct"] = (
                        round(p_profit / p_staked * 100, 2) if p_staked > 0 else 0.0
                    )
                    result["parlay_df"] = pd.DataFrame(p_rows)
        except Exception as e:
            result["parlay_error"] = str(e)

    return result


# ── Multi-day backtest ────────────────────────────────────────────────────────

def run_backtest(
    end_date: date,
    n_days: int,
    api_key: str,
    min_edge_pct: float = 2.0,
    home_field_adv: float = 0.06,
    season: int | None = None,
    build_parlays: bool = False,
    parlay_legs: int = 3,
    parlay_top_n: int = 8,
    bankroll: float = 10.0,
    max_parlays: int = 5,
    fractional_kelly: float = 0.35,
    progress_callback=None,
) -> tuple[list[dict], dict]:
    """
    Simulate the past n_days ending on end_date (inclusive).

    progress_callback(current_step, total_steps, current_date) is called each day.

    Returns (day_results, summary).
    """
    start_date = end_date - timedelta(days=n_days - 1)
    active_season = int(season or end_date.year)

    stats_service = StatsService()
    stats_df = stats_service.fetch_mlb_team_stats(active_season)

    day_results: list[dict] = []
    current = start_date
    step = 0

    while current <= end_date:
        step += 1
        if progress_callback:
            progress_callback(step, n_days, current)

        day_result = simulate_day(
            target_date=current,
            api_key=api_key,
            stats_df=stats_df,
            min_edge_pct=min_edge_pct,
            home_field_adv=home_field_adv,
            build_parlays=build_parlays,
            parlay_legs=parlay_legs,
            parlay_top_n=parlay_top_n,
            bankroll=bankroll,
            max_parlays=max_parlays,
            fractional_kelly=fractional_kelly,
        )
        day_results.append(day_result)

        # Abort immediately if the API rejects access — no point hammering N more days.
        if day_result.get("auth_error"):
            raise HistoricalOddsUnavailable(day_result["reason"])

        current += timedelta(days=1)

    # ── Summary ───────────────────────────────────────────────────────────────
    active = [r for r in day_results if not r.get("skipped")]
    total_picks = sum(r["picks"] for r in active)
    total_wins = sum(r["wins"] for r in active)
    total_losses = sum(r["losses"] for r in active)
    total_profit = sum(r["profit_u"] for r in active)
    active_days = len(active)

    # Running cumulative profit for each active day (for sparkline / chart)
    running_profit = []
    cum = 0.0
    for r in day_results:
        if not r.get("skipped"):
            cum += r["profit_u"]
        running_profit.append({"date": r["date"], "cumulative_profit_u": round(cum, 3)})

    summary: dict = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "n_days_requested": n_days,
        "active_days": active_days,
        "skipped_days": n_days - active_days,
        "min_edge_pct": min_edge_pct,
        "total_picks": total_picks,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "win_rate_pct": round(total_wins / total_picks * 100, 1) if total_picks else 0.0,
        "total_profit_u": round(total_profit, 3),
        "total_roi_pct": round(total_profit / total_picks * 100, 2) if total_picks else 0.0,
        "avg_profit_per_active_day": round(total_profit / active_days, 3) if active_days else 0.0,
        "running_profit": running_profit,
    }

    if build_parlays:
        p_active = [r for r in active if "parlay_count" in r]
        if p_active:
            total_p_staked = sum(r.get("parlay_staked", 0.0) for r in p_active)
            total_p_profit = sum(r.get("parlay_profit", 0.0) for r in p_active)
            total_p_wins = sum(r.get("parlay_wins", 0) for r in p_active)
            total_p_count = sum(r.get("parlay_count", 0) for r in p_active)
            summary["parlay_active_days"] = len(p_active)
            summary["total_parlay_count"] = total_p_count
            summary["total_parlay_wins"] = total_p_wins
            summary["total_parlay_staked"] = round(total_p_staked, 2)
            summary["total_parlay_profit"] = round(total_p_profit, 3)
            summary["total_parlay_roi_pct"] = (
                round(total_p_profit / total_p_staked * 100, 2) if total_p_staked > 0 else 0.0
            )

    return day_results, summary
