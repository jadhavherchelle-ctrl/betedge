from gambling_app.ui import BettingApp


if __name__ == "__main__":
    BettingApp().run()
    raise SystemExit

import itertools
import os
import re
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
MLB_ODDS_SPORT = "baseball_mlb"
NBA_ODDS_SPORT = "basketball_nba"
NBA_TEAM_NAMES = [
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "Los Angeles Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards",
]


@st.cache_data(ttl=60 * 5)
def fetch_live_mlb_odds(api_key: str) -> list[dict]:
    url = f"{ODDS_API_BASE}/sports/{MLB_ODDS_SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60 * 5)
def fetch_live_nba_odds(api_key: str) -> list[dict]:
    url = f"{ODDS_API_BASE}/sports/{NBA_ODDS_SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60 * 60)
def fetch_team_stat_profile(team_id: int, season: int) -> dict:
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
    params = {
        "stats": "season",
        "group": "hitting,pitching",
        "season": season,
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    payload = response.json()
    stats_map: dict[str, dict] = {}
    for stat_group in payload.get("stats", []):
        group_name = stat_group.get("group", {}).get("displayName", "").lower()
        splits = stat_group.get("splits", [])
        if not splits:
            continue
        stats_map[group_name] = splits[0].get("stat", {})

    return stats_map


@st.cache_data(ttl=60 * 60)
def fetch_mlb_team_stats(season: int) -> pd.DataFrame:
    # MLB standings endpoint
    url = "https://statsapi.mlb.com/api/v1/standings"
    params = {
        "leagueId": "103,104",
        "season": season,
        "standingsTypes": "regularSeason",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()
    rows = []

    for record_group in data.get("records", []):
        for team_record in record_group.get("teamRecords", []):
            team = team_record.get("team", {})
            team_name = team.get("name")
            team_id = team.get("id")
            wins = team_record.get("wins", 0)
            losses = team_record.get("losses", 0)
            games = wins + losses
            win_pct = wins / games if games else 0.5

            runs_scored = team_record.get("runsScored")
            runs_allowed = team_record.get("runsAllowed")
            run_diff = None
            if runs_scored is not None and runs_allowed is not None:
                run_diff = runs_scored - runs_allowed

            profile = fetch_team_stat_profile(int(team_id), season) if team_id is not None else {}
            hitting = profile.get("hitting", {})
            pitching = profile.get("pitching", {})

            hitting_games = float(hitting.get("gamesPlayed") or 0)
            pitching_games = float(pitching.get("gamesPlayed") or 0)
            hitting_runs = float(hitting.get("runs") or 0)
            pitching_runs = float(pitching.get("runs") or 0)

            hitting_ops = hitting.get("ops")
            pitching_era = pitching.get("era")
            pitching_whip = pitching.get("whip")

            try:
                hitting_ops = float(hitting_ops) if hitting_ops is not None else None
            except Exception:
                hitting_ops = None
            try:
                pitching_era = float(pitching_era) if pitching_era is not None else None
            except Exception:
                pitching_era = None
            try:
                pitching_whip = float(pitching_whip) if pitching_whip is not None else None
            except Exception:
                pitching_whip = None

            rows.append(
                {
                    "team_id": team_id,
                    "team": team_name,
                    "wins": wins,
                    "losses": losses,
                    "win_pct": win_pct,
                    "run_diff": run_diff,
                    "hitting_ops": hitting_ops,
                    "hitting_runs_per_game": (hitting_runs / hitting_games) if hitting_games else None,
                    "pitching_era": pitching_era,
                    "pitching_whip": pitching_whip,
                    "pitching_runs_allowed_per_game": (pitching_runs / pitching_games) if pitching_games else None,
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset=["team"])

    # Fallback in case API shape changes
    if df.empty:
        raise ValueError("Could not load MLB standings data.")

    def zscore(series: pd.Series) -> pd.Series:
        clean = pd.to_numeric(series, errors="coerce")
        std = clean.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0, index=series.index)
        return (clean.fillna(clean.mean()) - clean.mean()) / std

    # Better MLB power score: blend standings, run differential, offense, and pitching.
    score = (
        zscore(df["win_pct"]) * 0.20
        + zscore(df["run_diff"]) * 0.20
        + zscore(df["hitting_ops"]) * 0.20
        + zscore(df["hitting_runs_per_game"]) * 0.10
        + zscore(-df["pitching_era"]) * 0.20
        + zscore(-df["pitching_whip"]) * 0.05
        + zscore(-df["pitching_runs_allowed_per_game"]) * 0.05
    )

    df["power_score"] = score

    return df


def extract_nba_team_name(raw: str) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    for team_name in NBA_TEAM_NAMES:
        if text.endswith(team_name):
            return team_name
    match = re.search(r"([A-Z][A-Za-z0-9 .'-]+)$", text)
    if match:
        return match.group(1).strip()
    return text


@st.cache_data(ttl=60 * 30)
def fetch_nba_team_stats() -> pd.DataFrame:
    url = "https://www.espn.com/nba/standings"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    if len(tables) < 4:
        raise ValueError("Could not load NBA standings data.")

    rows = []
    conference_pairs = [
        (tables[0], tables[1], "East"),
        (tables[2], tables[3], "West"),
    ]

    for names_table, stats_table, conference in conference_pairs:
        team_names = [extract_nba_team_name(value) for value in names_table.iloc[:, 0].tolist()]
        stats_table = stats_table.copy().reset_index(drop=True)
        stats_table.columns = [str(col).strip() for col in stats_table.columns]

        for team_name, (_, row) in zip(team_names, stats_table.iterrows()):
            if not team_name:
                continue

            def as_float(value):
                try:
                    return float(value)
                except Exception:
                    return None

            wins = int(row.get("W", 0))
            losses = int(row.get("L", 0))
            games = wins + losses
            win_pct = as_float(row.get("PCT"))
            if win_pct is None:
                win_pct = wins / games if games else 0.5

            diff = as_float(row.get("DIFF"))
            ppg = as_float(row.get("PPG"))
            opp_ppg = as_float(row.get("OPP PPG"))

            l10 = str(row.get("L10", "")).strip()
            l10_pct = None
            if "-" in l10:
                try:
                    l10_w, l10_l = l10.split("-", 1)
                    l10_w = float(l10_w)
                    l10_l = float(l10_l)
                    l10_total = l10_w + l10_l
                    l10_pct = (l10_w / l10_total) if l10_total else None
                except Exception:
                    l10_pct = None

            rows.append(
                {
                    "team": team_name,
                    "conference": conference,
                    "wins": wins,
                    "losses": losses,
                    "win_pct": win_pct,
                    "point_diff": diff,
                    "ppg": ppg,
                    "opp_ppg": opp_ppg,
                    "l10_win_pct": l10_pct,
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset=["team"])

    if df.empty:
        raise ValueError("Could not load NBA standings data.")

    def zscore(series: pd.Series) -> pd.Series:
        clean = pd.to_numeric(series, errors="coerce")
        std = clean.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0, index=series.index)
        return (clean.fillna(clean.mean()) - clean.mean()) / std

    score = (
        zscore(df["win_pct"]) * 0.25
        + zscore(df["point_diff"]) * 0.25
        + zscore(df["ppg"]) * 0.20
        + zscore(-df["opp_ppg"]) * 0.20
        + zscore(df["l10_win_pct"]) * 0.10
    )

    df["power_score"] = score
    return df


def normalize_team_name(name: str) -> str:
    mapping = {
        "Arizona Diamondbacks": "D-backs",
        "D-backs": "D-backs",
        "Atlanta Braves": "Braves",
        "Braves": "Braves",
        "Athletics": "Athletics",
        "Baltimore Orioles": "Orioles",
        "Orioles": "Orioles",
        "Boston Red Sox": "Red Sox",
        "Red Sox": "Red Sox",
        "Chicago Cubs": "Cubs",
        "Cubs": "Cubs",
        "Chicago White Sox": "White Sox",
        "White Sox": "White Sox",
        "Cincinnati Reds": "Reds",
        "Reds": "Reds",
        "Cleveland Guardians": "Guardians",
        "Guardians": "Guardians",
        "Colorado Rockies": "Rockies",
        "Rockies": "Rockies",
        "Detroit Tigers": "Tigers",
        "Tigers": "Tigers",
        "Houston Astros": "Astros",
        "Astros": "Astros",
        "Kansas City Royals": "Royals",
        "Royals": "Royals",
        "Los Angeles Angels": "Angels",
        "Angels": "Angels",
        "Los Angeles Dodgers": "Dodgers",
        "Dodgers": "Dodgers",
        "Miami Marlins": "Marlins",
        "Marlins": "Marlins",
        "Milwaukee Brewers": "Brewers",
        "Brewers": "Brewers",
        "Minnesota Twins": "Twins",
        "Twins": "Twins",
        "New York Mets": "Mets",
        "Mets": "Mets",
        "New York Yankees": "Yankees",
        "Yankees": "Yankees",
        "Philadelphia Phillies": "Phillies",
        "Phillies": "Phillies",
        "Pittsburgh Pirates": "Pirates",
        "Pirates": "Pirates",
        "San Diego Padres": "Padres",
        "Padres": "Padres",
        "San Francisco Giants": "Giants",
        "Giants": "Giants",
        "Seattle Mariners": "Mariners",
        "Mariners": "Mariners",
        "St. Louis Cardinals": "Cardinals",
        "Cardinals": "Cardinals",
        "Tampa Bay Rays": "Rays",
        "Rays": "Rays",
        "Texas Rangers": "Rangers",
        "Rangers": "Rangers",
        "Toronto Blue Jays": "Blue Jays",
        "Blue Jays": "Blue Jays",
        "Washington Nationals": "Nationals",
        "Nationals": "Nationals",
    }
    return mapping.get(name, name)


def normalize_nba_team_name(name: str) -> str:
    mapping = {
        "Atlanta Hawks": "Atlanta Hawks",
        "Boston Celtics": "Boston Celtics",
        "Brooklyn Nets": "Brooklyn Nets",
        "Charlotte Hornets": "Charlotte Hornets",
        "Chicago Bulls": "Chicago Bulls",
        "Cleveland Cavaliers": "Cleveland Cavaliers",
        "Dallas Mavericks": "Dallas Mavericks",
        "Denver Nuggets": "Denver Nuggets",
        "Detroit Pistons": "Detroit Pistons",
        "Golden State Warriors": "Golden State Warriors",
        "Houston Rockets": "Houston Rockets",
        "Indiana Pacers": "Indiana Pacers",
        "LA Clippers": "Los Angeles Clippers",
        "L.A. Clippers": "Los Angeles Clippers",
        "Los Angeles Clippers": "Los Angeles Clippers",
        "Los Angeles Lakers": "Los Angeles Lakers",
        "Memphis Grizzlies": "Memphis Grizzlies",
        "Miami Heat": "Miami Heat",
        "Milwaukee Bucks": "Milwaukee Bucks",
        "Minnesota Timberwolves": "Minnesota Timberwolves",
        "New Orleans Pelicans": "New Orleans Pelicans",
        "New York Knicks": "New York Knicks",
        "Oklahoma City Thunder": "Oklahoma City Thunder",
        "Orlando Magic": "Orlando Magic",
        "Philadelphia 76ers": "Philadelphia 76ers",
        "Phoenix Suns": "Phoenix Suns",
        "Portland Trail Blazers": "Portland Trail Blazers",
        "Sacramento Kings": "Sacramento Kings",
        "San Antonio Spurs": "San Antonio Spurs",
        "Toronto Raptors": "Toronto Raptors",
        "Utah Jazz": "Utah Jazz",
        "Washington Wizards": "Washington Wizards",
    }
    return mapping.get(name, name)


def american_to_decimal(american_odds: float) -> float:
    if american_odds > 0:
        return 1 + (american_odds / 100)
    return 1 + (100 / abs(american_odds))


def american_to_implied_prob(american_odds: float) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def get_best_h2h_prices(game: dict, normalizer=normalize_team_name) -> dict:
    home_team = normalizer(game.get("home_team"))
    away_team = normalizer(game.get("away_team"))

    best = {
        home_team: {"price": None, "book": None},
        away_team: {"price": None, "book": None},
    }

    for bookmaker in game.get("bookmakers", []):
        book_name = bookmaker.get("title", "Unknown")
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                team = normalizer(outcome.get("name"))
                price = outcome.get("price")
                if team not in best or price is None:
                    continue

                # choose best payout price (higher american odds)
                current = best[team]["price"]
                if current is None or price > current:
                    best[team] = {"price": price, "book": book_name}

    return best


def get_market_consensus_h2h(game: dict, normalizer=normalize_team_name) -> dict:
    home_team = normalizer(game.get("home_team"))
    away_team = normalizer(game.get("away_team"))

    consensus_rows = []
    for bookmaker in game.get("bookmakers", []):
        outcomes = {}
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                team = normalizer(outcome.get("name"))
                price = outcome.get("price")
                if team in {home_team, away_team} and price is not None:
                    outcomes[team] = american_to_implied_prob(price)

        if home_team in outcomes and away_team in outcomes:
            total = outcomes[home_team] + outcomes[away_team]
            if total > 0:
                consensus_rows.append(
                    {
                        home_team: outcomes[home_team] / total,
                        away_team: outcomes[away_team] / total,
                    }
                )

    if not consensus_rows:
        return {home_team: 0.5, away_team: 0.5, "book_count": 0}

    home_prob = float(np.mean([row[home_team] for row in consensus_rows]))
    away_prob = 1 - home_prob

    return {home_team: home_prob, away_team: away_prob, "book_count": len(consensus_rows)}


def logistic(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def build_game_edges(
    odds_games: list[dict],
    stats_df: pd.DataFrame,
    home_field_adv: float,
    normalizer=normalize_team_name,
) -> pd.DataFrame:
    team_power = {normalizer(r["team"]): r["power_score"] for _, r in stats_df.iterrows()}

    rows = []

    for game in odds_games:
        home = normalizer(game.get("home_team"))
        away = normalizer(game.get("away_team"))

        if home not in team_power or away not in team_power:
            continue

        best_prices = get_best_h2h_prices(game, normalizer=normalizer)

        if best_prices.get(home, {}).get("price") is None or best_prices.get(away, {}).get("price") is None:
            continue

        home_price = best_prices[home]["price"]
        away_price = best_prices[away]["price"]

        market = get_market_consensus_h2h(game, normalizer=normalizer)
        market_home_prob = market.get(home, 0.5)
        market_away_prob = market.get(away, 0.5)

        home_implied = american_to_implied_prob(home_price)
        away_implied = american_to_implied_prob(away_price)

        # Remove vig by normalizing implied probabilities
        implied_sum = home_implied + away_implied
        if implied_sum <= 0:
            continue

        home_implied_no_vig = home_implied / implied_sum
        away_implied_no_vig = away_implied / implied_sum

        power_diff = team_power[home] - team_power[away] + home_field_adv
        stat_model_prob = float(logistic(power_diff))
        home_model_prob = (stat_model_prob * 0.72) + (market_home_prob * 0.28)
        home_model_prob = float(np.clip(home_model_prob, 0.01, 0.99))
        away_model_prob = 1 - home_model_prob

        commence = game.get("commence_time")
        try:
            start_time = datetime.fromisoformat(commence.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            start_time = None

        rows.extend(
            [
                {
                    "game": f"{away} @ {home}",
                    "start_utc": start_time,
                    "team": home,
                    "opponent": away,
                    "side": "home",
                    "book": best_prices[home]["book"],
                    "books": market.get("book_count", 0),
                    "american_odds": home_price,
                    "decimal_odds": american_to_decimal(home_price),
                    "implied_prob_no_vig": market_home_prob,
                    "model_prob": home_model_prob,
                    "edge": home_model_prob - market_home_prob,
                },
                {
                    "game": f"{away} @ {home}",
                    "start_utc": start_time,
                    "team": away,
                    "opponent": home,
                    "side": "away",
                    "book": best_prices[away]["book"],
                    "books": market.get("book_count", 0),
                    "american_odds": away_price,
                    "decimal_odds": american_to_decimal(away_price),
                    "implied_prob_no_vig": market_away_prob,
                    "model_prob": away_model_prob,
                    "edge": away_model_prob - market_away_prob,
                },
            ]
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="edge", ascending=False)
    return df


def build_parlays(plays_df: pd.DataFrame, max_legs: int, top_n: int) -> pd.DataFrame:
    picks = plays_df.head(top_n).copy()
    if picks.empty:
        return pd.DataFrame()

    parlays = []

    for legs in range(2, max_legs + 1):
        for combo in itertools.combinations(picks.to_dict("records"), legs):
            # Avoid same game in one parlay
            games = [c["game"] for c in combo]
            if len(games) != len(set(games)):
                continue

            dec_odds = np.prod([c["decimal_odds"] for c in combo])
            model_prob = np.prod([c["model_prob"] for c in combo])
            implied_prob = 1 / dec_odds
            edge = model_prob - implied_prob
            ev = model_prob * dec_odds - 1

            parlays.append(
                {
                    "legs": legs,
                    "parlay": " + ".join([f"{c['team']} ({c['american_odds']:+})" for c in combo]),
                    "games": tuple(games),
                    "teams": tuple(c["team"] for c in combo),
                    "decimal_odds": dec_odds,
                    "implied_prob": implied_prob,
                    "model_prob": model_prob,
                    "edge": edge,
                    "ev_per_1u": ev,
                }
            )

    parlay_df = pd.DataFrame(parlays)
    if not parlay_df.empty:
        parlay_df = parlay_df.sort_values(by="ev_per_1u", ascending=False)
    return parlay_df


def _parlay_overlap_score(candidate: pd.Series, selected: list[pd.Series]) -> int:
    candidate_games = set(candidate["games"])
    candidate_teams = set(candidate["teams"])

    overlap = 0
    for parlay in selected:
        overlap += len(candidate_games.intersection(set(parlay["games"])))
        overlap += len(candidate_teams.intersection(set(parlay["teams"])))
    return overlap


def select_diversified_parlays(
    parlay_df: pd.DataFrame,
    max_parlays: int,
    diversification_bias: float,
    max_team_uses: int,
) -> pd.DataFrame:
    if parlay_df.empty:
        return pd.DataFrame()

    scored = parlay_df.copy()
    scored = scored[scored["ev_per_1u"] > 0].copy()
    if scored.empty:
        return pd.DataFrame()

    scored["unique_teams"] = scored["teams"].apply(lambda x: len(set(x)))
    scored["base_score"] = scored["ev_per_1u"] * (1 + (scored["unique_teams"] * 0.06))
    scored = scored.sort_values(by="base_score", ascending=False)

    selected: list[pd.Series] = []
    team_use_counts: dict[str, int] = {}
    for _, row in scored.iterrows():
        if len(selected) >= max_parlays:
            break

        teams = list(row["teams"])
        if any(team_use_counts.get(team, 0) >= max_team_uses for team in teams):
            continue

        overlap = _parlay_overlap_score(row, selected)
        adjusted_score = float(row["base_score"]) / (1 + (diversification_bias * overlap))
        if adjusted_score <= 0:
            continue

        row = row.copy()
        row["adjusted_score"] = adjusted_score
        row["overlap_score"] = overlap
        selected.append(row)

        for team in teams:
            team_use_counts[team] = team_use_counts.get(team, 0) + 1

    if not selected:
        return pd.DataFrame()

    selected_df = pd.DataFrame(selected)
    selected_df = selected_df.sort_values(by="adjusted_score", ascending=False)
    return selected_df


def allocate_bankroll_to_parlays(
    parlay_df: pd.DataFrame,
    bankroll: float,
    max_parlays: int,
    max_bet_pct: float,
    fractional_kelly: float,
    diversification_bias: float,
    max_team_uses: int,
) -> pd.DataFrame:
    if parlay_df.empty or bankroll <= 0:
        return pd.DataFrame()

    candidates = select_diversified_parlays(
        parlay_df,
        max_parlays=max_parlays,
        diversification_bias=diversification_bias,
        max_team_uses=max_team_uses,
    )
    if candidates.empty:
        return pd.DataFrame()

    # Stake weights favor higher edge, but stay diversified and conservative.
    weights = candidates["ev_per_1u"].clip(lower=0).copy()
    weights = weights * (1 + (candidates["unique_teams"] * 0.03))
    # Lower fractional Kelly values flatten the curve so money is spread across more parlays.
    shape_exponent = 0.45 + (0.55 * max(min(fractional_kelly, 1.0), 0.0))
    weights = weights.pow(shape_exponent)

    if float(weights.sum()) <= 0:
        return pd.DataFrame()

    kelly_fraction = weights / float(weights.sum())
    kelly_fraction = kelly_fraction.clip(lower=0, upper=max_bet_pct)

    if float(kelly_fraction.sum()) <= 0:
        return pd.DataFrame()

    total_fraction = float(kelly_fraction.sum())
    if total_fraction > 1:
        kelly_fraction = kelly_fraction / total_fraction

    stake = bankroll * kelly_fraction

    out = candidates.copy()
    out["kelly_fraction"] = kelly_fraction
    out["stake"] = stake
    out = out[out["stake"] > 0].copy()

    if out.empty:
        return pd.DataFrame()

    out["potential_payout"] = out["stake"] * out["decimal_odds"]
    out["potential_profit"] = out["potential_payout"] - out["stake"]
    out["expected_profit"] = out["stake"] * out["ev_per_1u"]
    out["win_loss_pct"] = out["model_prob"] * 100
    out["loss_pct"] = (1 - out["model_prob"]) * 100
    out["team_count"] = out["teams"].apply(lambda teams: len(set(teams)))
    out = out.sort_values(by="expected_profit", ascending=False)

    return out


def summarize_allocation_risk(alloc_df: pd.DataFrame) -> dict:
    if alloc_df.empty:
        return {
            "lose_all_prob": 0.0,
            "at_least_one_win_prob": 0.0,
            "total_stake": 0.0,
            "total_payout_if_all_win": 0.0,
            "total_profit_if_all_win": 0.0,
            "expected_profit": 0.0,
        }

    win_probs = alloc_df["model_prob"].clip(lower=0, upper=1)
    lose_all_prob = float(np.prod(1 - win_probs))
    at_least_one_win_prob = 1 - lose_all_prob
    total_stake = float(alloc_df["stake"].sum())
    total_payout_if_all_win = float(alloc_df["potential_payout"].sum())
    total_profit_if_all_win = float(alloc_df["potential_profit"].sum())
    expected_profit = float(alloc_df["expected_profit"].sum())

    return {
        "lose_all_prob": lose_all_prob,
        "at_least_one_win_prob": at_least_one_win_prob,
        "total_stake": total_stake,
        "total_payout_if_all_win": total_payout_if_all_win,
        "total_profit_if_all_win": total_profit_if_all_win,
        "expected_profit": expected_profit,
    }


def filter_later_today(edges_df: pd.DataFrame) -> pd.DataFrame:
    if edges_df.empty or "start_utc" not in edges_df.columns:
        return edges_df

    out = edges_df.copy()
    out["start_utc"] = pd.to_datetime(out["start_utc"], utc=True, errors="coerce")

    local_now = datetime.now().astimezone()
    local_tz = local_now.tzinfo
    if local_tz is None:
        return out

    start_utc = local_now.astimezone(timezone.utc)
    end_of_day_local = local_now.replace(hour=23, minute=59, second=59, microsecond=999999)
    end_utc = end_of_day_local.astimezone(timezone.utc)

    mask = (out["start_utc"].notna()) & (out["start_utc"] >= start_utc) & (out["start_utc"] <= end_utc)
    return out[mask].copy()


def render_league_section(
    league_label: str,
    edges_df: pd.DataFrame,
    min_edge: float,
    only_later_today: bool,
    bankroll: float,
    max_alloc_parlays: int,
    fractional_kelly: float,
    diversification_bias: float,
    max_team_uses: int,
    max_bet_pct: float,
    parlay_legs: int,
    parlay_top_n: int,
) -> None:
    st.subheader(f"{league_label} games in scope")

    if edges_df.empty:
        st.info(f"No {league_label.lower()} moneyline games found right now.")
        return

    if only_later_today:
        edges_df = filter_later_today(edges_df)
        if edges_df.empty:
            st.info(f"No {league_label.lower()} games later today in your local timezone.")
            return

    schedule = (
        edges_df[["game", "start_utc"]]
        .drop_duplicates(subset=["game"])
        .sort_values(by="start_utc")
        .copy()
    )
    local_tz = datetime.now().astimezone().tzinfo
    schedule["start_local"] = pd.to_datetime(schedule["start_utc"], utc=True).dt.tz_convert(local_tz)
    schedule["start_local"] = schedule["start_local"].dt.strftime("%Y-%m-%d %I:%M %p %Z")
    st.write(f"{len(schedule)} games loaded")
    st.dataframe(
        schedule[["game", "start_local"]].rename(columns={"game": "Game", "start_local": "Start (Local)"}),
        width="stretch",
    )

    filtered = edges_df[edges_df["edge"] >= (min_edge / 100)].copy()

    st.subheader("Best straight bet candidates")
    st.write(f"Found {len(filtered)} picks with edge ≥ {min_edge:.1f}%")

    if filtered.empty:
        st.info("No picks pass your edge filter. Lower the minimum edge.")
        return

    display_cols = ["game", "team", "book", "books", "american_odds", "model_prob", "implied_prob_no_vig", "edge"]
    table = filtered[display_cols].copy()
    table["model_prob"] = (table["model_prob"] * 100).round(2)
    table["implied_prob_no_vig"] = (table["implied_prob_no_vig"] * 100).round(2)
    table["edge"] = (table["edge"] * 100).round(2)
    table.columns = ["Game", "Team", "Book", "Books", "Odds", "Model %", "Market % (No Vig)", "Edge %"]
    st.dataframe(table, width="stretch")

    st.subheader("Parlay ideas")
    parlay_df = build_parlays(filtered, max_legs=parlay_legs, top_n=parlay_top_n)

    if parlay_df.empty:
        st.info("Not enough qualifying picks to build parlays.")
        return

    display_out = parlay_df[["legs", "parlay", "decimal_odds", "implied_prob", "model_prob", "edge", "ev_per_1u"]].copy()
    display_out["decimal_odds"] = display_out["decimal_odds"].round(2)
    display_out["implied_prob"] = (display_out["implied_prob"] * 100).round(2)
    display_out["model_prob"] = (display_out["model_prob"] * 100).round(2)
    display_out["edge"] = (display_out["edge"] * 100).round(2)
    display_out["ev_per_1u"] = display_out["ev_per_1u"].round(3)
    display_out.columns = ["Legs", "Parlay", "Decimal Odds", "Implied %", "Model %", "Edge %", "EV / 1u"]
    st.dataframe(display_out.head(30), width="stretch")

    st.subheader("Bankroll allocation plan")
    alloc_df = allocate_bankroll_to_parlays(
        parlay_df=parlay_df,
        bankroll=float(bankroll),
        max_parlays=int(max_alloc_parlays),
        max_bet_pct=float(max_bet_pct) / 100,
        fractional_kelly=float(fractional_kelly),
        diversification_bias=float(diversification_bias),
        max_team_uses=int(max_team_uses),
    )

    if alloc_df.empty:
        st.info("No positive-EV parlays available for allocation with current settings.")
        return

    risk = summarize_allocation_risk(alloc_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bankroll", f"${bankroll:,.2f}")
    c2.metric("Allocated", f"${risk['total_stake']:,.2f}")
    c3.metric("Expected profit", f"${risk['expected_profit']:,.2f}")
    c4.metric("Lose everything", f"{risk['lose_all_prob'] * 100:.2f}%")

    summary_cols = st.columns(3)
    summary_cols[0].metric("Chance of at least one win", f"{risk['at_least_one_win_prob'] * 100:.2f}%")
    summary_cols[1].metric("Max payout if all win", f"${risk['total_payout_if_all_win']:,.2f}")
    summary_cols[2].metric("Max profit if all win", f"${risk['total_profit_if_all_win']:,.2f}")

    st.caption("Win/loss percentages use the model probability for each parlay. Losing everything is an approximate combined probability assuming the selected parlays are independent.")

    team_exposure = (
        alloc_df.assign(team_list=alloc_df["teams"])
        .explode("team_list")
        .groupby("team_list", as_index=False)
        .agg(parlay_count=("team_list", "size"), total_stake=("stake", "sum"))
        .sort_values(by=["parlay_count", "total_stake"], ascending=False)
    )
    team_exposure.columns = ["Team", "Parlay Count", "Total Stake ($)"]
    st.subheader("Team exposure")
    st.dataframe(team_exposure, width="stretch")

    display_alloc = alloc_df[
        [
            "legs",
            "parlay",
            "stake",
            "kelly_fraction",
            "win_loss_pct",
            "loss_pct",
            "decimal_odds",
            "potential_payout",
            "potential_profit",
            "expected_profit",
            "ev_per_1u",
            "team_count",
        ]
    ].copy()
    display_alloc["kelly_fraction"] = (display_alloc["kelly_fraction"] * 100).round(2)
    display_alloc["win_loss_pct"] = display_alloc["win_loss_pct"].round(2)
    display_alloc["loss_pct"] = display_alloc["loss_pct"].round(2)
    display_alloc["stake"] = display_alloc["stake"].round(2)
    display_alloc["decimal_odds"] = display_alloc["decimal_odds"].round(2)
    display_alloc["potential_payout"] = display_alloc["potential_payout"].round(2)
    display_alloc["potential_profit"] = display_alloc["potential_profit"].round(2)
    display_alloc["expected_profit"] = display_alloc["expected_profit"].round(2)
    display_alloc["ev_per_1u"] = display_alloc["ev_per_1u"].round(3)
    display_alloc["team_count"] = display_alloc["team_count"].astype(int)
    display_alloc.columns = [
        "Legs",
        "Parlay",
        "Stake ($)",
        "Kelly %",
        "Win %",
        "Loss %",
        "Decimal Odds",
        "Could Win ($)",
        "Max Profit ($)",
        "Expected Profit ($)",
        "EV / $1",
        "Teams",
    ]
    st.dataframe(display_alloc, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Herchelles ⚾ MLB Live Odds + Best Bet/Parlay Finder", layout="wide")
    st.title("Herchelles ⚾ MLB Live Odds + Best Bet/Parlay Finder")
    st.caption("Educational model only — not financial advice.")

    secrets_key = ""
    try:
        secrets_key = st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        secrets_key = ""

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "The Odds API key",
            value=os.getenv("ODDS_API_KEY", secrets_key),
            type="password",
            help="Get a key from the-odds-api.com and set ODDS_API_KEY or paste it here.",
        )

        season = st.number_input("Season", min_value=2020, max_value=2100, value=datetime.now().year)
        min_edge = st.slider("Minimum edge (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
        home_field_adv = st.slider(
            "Home field adjustment", min_value=-0.25, max_value=0.25, value=0.06, step=0.01,
            help="Positive values give home teams a model boost.",
        )
        parlay_legs = st.selectbox("Max parlay legs", options=[2, 3, 4], index=1)
        parlay_top_n = st.slider("Use top N picks for parlays", min_value=4, max_value=14, value=8)
        bankroll = st.number_input("Bankroll ($)", min_value=1.0, value=10.0, step=1.0)
        max_alloc_parlays = st.slider("Allocate across top parlays", min_value=2, max_value=20, value=10)
        fractional_kelly = st.slider(
            "Fractional Kelly", min_value=0.10, max_value=1.0, value=0.35, step=0.05,
            help="Lower values spread money out more and reduce risk.",
        )
        diversification_bias = st.slider(
            "Diversification bias", min_value=0.0, max_value=2.0, value=0.85, step=0.05,
            help="Higher values penalize parlays that reuse the same teams or games.",
        )
        max_team_uses = st.slider(
            "Max uses per team", min_value=1, max_value=6, value=2,
            help="Limits how many selected parlays can include the same team.",
        )
        max_bet_pct = st.slider(
            "Max stake per parlay (% of bankroll)",
            min_value=5.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
        )
        only_later_today = st.checkbox("Only games later today", value=True)

    if not api_key:
        st.warning("Add your The Odds API key in the sidebar to load live odds.")
        st.stop()

    with st.spinner("Loading live odds + stats for baseball and basketball..."):
        baseball_edges = pd.DataFrame()
        basketball_edges = pd.DataFrame()

        try:
            baseball_odds = fetch_live_mlb_odds(api_key)
            baseball_stats = fetch_mlb_team_stats(int(season))
            baseball_edges = build_game_edges(baseball_odds, baseball_stats, home_field_adv, normalizer=normalize_team_name)
        except requests.HTTPError as err:
            st.error(f"Baseball API error: {err}")
        except Exception as err:
            st.error(f"Baseball error: {err}")

        try:
            basketball_odds = fetch_live_nba_odds(api_key)
            basketball_stats = fetch_nba_team_stats()
            basketball_edges = build_game_edges(
                basketball_odds,
                basketball_stats,
                home_field_adv,
                normalizer=normalize_nba_team_name,
            )
        except requests.HTTPError as err:
            st.error(f"Basketball API error: {err}")
        except Exception as err:
            st.error(f"Basketball error: {err}")

    baseball_tab, basketball_tab = st.tabs(["Baseball", "Basketball"])

    with baseball_tab:
        render_league_section(
            league_label="Baseball",
            edges_df=baseball_edges,
            min_edge=min_edge,
            only_later_today=only_later_today,
            bankroll=bankroll,
            max_alloc_parlays=max_alloc_parlays,
            fractional_kelly=fractional_kelly,
            diversification_bias=diversification_bias,
            max_team_uses=max_team_uses,
            max_bet_pct=max_bet_pct,
            parlay_legs=parlay_legs,
            parlay_top_n=parlay_top_n,
        )

    with basketball_tab:
        render_league_section(
            league_label="Basketball",
            edges_df=basketball_edges,
            min_edge=min_edge,
            only_later_today=only_later_today,
            bankroll=bankroll,
            max_alloc_parlays=max_alloc_parlays,
            fractional_kelly=fractional_kelly,
            diversification_bias=diversification_bias,
            max_team_uses=max_team_uses,
            max_bet_pct=max_bet_pct,
            parlay_legs=parlay_legs,
            parlay_top_n=parlay_top_n,
        )


if __name__ == "__main__":
    main()
