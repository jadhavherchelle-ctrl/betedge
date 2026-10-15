from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests
import streamlit as st

from .constants import MLB_ODDS_SPORT, NBA_ODDS_SPORT, ODDS_API_BASE
from .constants import MLB_TEAM_NAME_MAP, NBA_TEAM_NAMES


@st.cache_data(ttl=60 * 5)
def fetch_live_odds(api_key: str, sport_key: str, regions: str = "us") -> list[dict]:
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60 * 2)
def fetch_scores(api_key: str, sport_key: str, days_from: int = 1) -> list[dict]:
    url = f"{ODDS_API_BASE}/sports/{sport_key}/scores"
    params = {
        "apiKey": api_key,
        "daysFrom": int(days_from),
        "dateFormat": "iso",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


@st.cache_data(ttl=60 * 60)
def fetch_available_sports(api_key: str) -> list[dict]:
    url = f"{ODDS_API_BASE}/sports"
    params = {"apiKey": api_key, "all": "true"}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


@st.cache_data(ttl=60 * 60)
def fetch_active_tennis_sport_keys(api_key: str) -> list[str]:
    sports = fetch_available_sports(api_key)
    keys = [
        sport.get("key", "")
        for sport in sports
        if sport.get("active") and sport.get("group") == "Tennis" and sport.get("key")
    ]
    return sorted(keys)


@st.cache_data(ttl=60 * 60)
def fetch_team_stat_profile(team_id: int, season: int) -> dict:
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
    params = {"stats": "season", "group": "hitting,pitching", "season": season}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    payload = response.json()
    stats_map: dict[str, dict] = {}
    for stat_group in payload.get("stats", []):
        group_name = stat_group.get("group", {}).get("displayName", "").lower()
        splits = stat_group.get("splits", [])
        if splits:
            stats_map[group_name] = splits[0].get("stat", {})
    return stats_map


@st.cache_data(ttl=60 * 30)
def fetch_pitcher_stat_profile(player_id: int, season: int) -> dict:
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
    params = {"stats": "season", "group": "pitching", "season": season}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    payload = response.json()
    stats = payload.get("stats", [])
    if not stats:
        return {}
    splits = stats[0].get("splits", [])
    if not splits:
        return {}
    return splits[0].get("stat", {})


def _normalize_mlb_name(name: str) -> str:
    return MLB_TEAM_NAME_MAP.get(name, name)


def _innings_to_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "." not in text:
        try:
            return float(text)
        except Exception:
            return None
    whole, frac = text.split(".", 1)
    try:
        whole_i = int(whole)
        frac_i = int(frac)
    except Exception:
        return None
    # MLB uses tenths as thirds of an inning (e.g., 5.2 = 5 and 2/3)
    return float(whole_i + (frac_i / 3))


@st.cache_data(ttl=60 * 15)
def fetch_team_bullpen_context(team_id: int, season: int, end_date: str) -> dict:
    end_dt = datetime.fromisoformat(end_date)
    start_dt = end_dt - pd.Timedelta(days=2)

    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
    params = {
        "stats": "byDateRange",
        "group": "pitching",
        "season": season,
        "startDate": start_dt.date().isoformat(),
        "endDate": end_dt.date().isoformat(),
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException:
        # Playoff schedules can list a placeholder "winner of series X" team
        # (e.g. "ATL/PHI") with a non-standard id that has no real stats
        # endpoint and 404s. Treat any fetch failure here as "no data"
        # rather than crashing the whole page.
        return {}

    stats = payload.get("stats", [])
    if not stats:
        return {}
    splits = stats[0].get("splits", [])
    if not splits:
        return {}

    stat = splits[0].get("stat", {})

    innings = _innings_to_float(stat.get("inningsPitched"))
    games_started = stat.get("gamesStarted")
    try:
        games_started = float(games_started) if games_started is not None else 0.0
    except Exception:
        games_started = 0.0

    # Approximate bullpen innings from team totals in last 3 days.
    bullpen_ip = None
    if innings is not None:
        bullpen_ip = max(0.0, float(innings) - (games_started * 5.2))

    def as_float(v):
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    return {
        "bullpen_ip_last3": bullpen_ip,
        "bullpen_era_last3": as_float(stat.get("era")),
        "bullpen_whip_last3": as_float(stat.get("whip")),
    }


@st.cache_data(ttl=60 * 10)
def fetch_game_lineup_context(game_pk: int) -> dict:
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()

    out = {}
    for side in ["home", "away"]:
        team = payload.get("teams", {}).get(side, {})
        batting_order = team.get("battingOrder", []) or []
        confirmed = len(batting_order) >= 9
        ops_values = []

        if confirmed:
            players = team.get("players", {})
            for player_id in batting_order[:9]:
                pdata = players.get(f"ID{player_id}", {})
                ops = (
                    pdata.get("seasonStats", {}).get("batting", {}).get("ops")
                    or pdata.get("stats", {}).get("batting", {}).get("ops")
                )
                try:
                    if ops is not None:
                        ops_values.append(float(ops))
                except Exception:
                    pass

        out[f"{side}_lineup_confirmed"] = confirmed
        out[f"{side}_lineup_ops"] = (sum(ops_values) / len(ops_values)) if ops_values else None

    return out


@st.cache_data(ttl=60 * 10)
def fetch_mlb_game_context(odds_games: list[dict], season: int) -> dict:
    game_dates = set()
    for game in odds_games:
        commence = game.get("commence_time")
        if not commence:
            continue
        try:
            dt = datetime.fromisoformat(commence.replace("Z", "+00:00")).astimezone(timezone.utc)
            game_dates.add(dt.date().isoformat())
        except Exception:
            continue

    schedule_games = []
    for date_str in sorted(game_dates):
        url = "https://statsapi.mlb.com/api/v1/schedule"
        params = {
            "sportId": 1,
            "date": date_str,
            "hydrate": "probablePitcher,weather,team",
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        for date_block in payload.get("dates", []):
            schedule_games.extend(date_block.get("games", []))

    pitcher_ids = set()
    matched = {}
    for g in schedule_games:
        home_raw = g.get("teams", {}).get("home", {}).get("team", {}).get("name")
        away_raw = g.get("teams", {}).get("away", {}).get("team", {}).get("name")
        home_team_id = g.get("teams", {}).get("home", {}).get("team", {}).get("id")
        away_team_id = g.get("teams", {}).get("away", {}).get("team", {}).get("id")
        home = _normalize_mlb_name(home_raw)
        away = _normalize_mlb_name(away_raw)
        key = f"{away} @ {home}"

        home_pitcher = g.get("teams", {}).get("home", {}).get("probablePitcher", {})
        away_pitcher = g.get("teams", {}).get("away", {}).get("probablePitcher", {})
        if home_pitcher.get("id"):
            pitcher_ids.add(int(home_pitcher["id"]))
        if away_pitcher.get("id"):
            pitcher_ids.add(int(away_pitcher["id"]))

        weather = g.get("weather", {}) or {}
        official_date = g.get("officialDate")
        lineup_ctx = fetch_game_lineup_context(int(g["gamePk"])) if g.get("gamePk") else {}

        home_bullpen = {}
        away_bullpen = {}
        if official_date and home_team_id:
            home_bullpen = fetch_team_bullpen_context(int(home_team_id), season, official_date)
        if official_date and away_team_id:
            away_bullpen = fetch_team_bullpen_context(int(away_team_id), season, official_date)

        matched[key] = {
            "home_pitcher_id": home_pitcher.get("id"),
            "home_pitcher_name": home_pitcher.get("fullName"),
            "away_pitcher_id": away_pitcher.get("id"),
            "away_pitcher_name": away_pitcher.get("fullName"),
            "weather_temp_f": weather.get("temp"),
            "weather_wind": weather.get("wind"),
            "weather_condition": weather.get("condition"),
            "home_bullpen": home_bullpen,
            "away_bullpen": away_bullpen,
            "home_lineup_confirmed": lineup_ctx.get("home_lineup_confirmed", False),
            "away_lineup_confirmed": lineup_ctx.get("away_lineup_confirmed", False),
            "home_lineup_ops": lineup_ctx.get("home_lineup_ops"),
            "away_lineup_ops": lineup_ctx.get("away_lineup_ops"),
        }

    pitcher_profiles = {}
    for pid in pitcher_ids:
        stat = fetch_pitcher_stat_profile(int(pid), season)

        def as_float(value):
            try:
                return float(value)
            except Exception:
                return None

        pitcher_profiles[int(pid)] = {
            "era": as_float(stat.get("era")),
            "whip": as_float(stat.get("whip")),
            "k_per_9": as_float(stat.get("strikeoutsPer9Inn")),
            "bb_per_9": as_float(stat.get("walksPer9Inn")),
            "k_bb": as_float(stat.get("strikeoutWalkRatio")),
            "innings_pitched": as_float(stat.get("inningsPitched")),
        }

    for game_key, ctx in matched.items():
        h_id = ctx.get("home_pitcher_id")
        a_id = ctx.get("away_pitcher_id")
        if h_id is not None:
            ctx["home_pitcher_stats"] = pitcher_profiles.get(int(h_id), {})
        else:
            ctx["home_pitcher_stats"] = {}
        if a_id is not None:
            ctx["away_pitcher_stats"] = pitcher_profiles.get(int(a_id), {})
        else:
            ctx["away_pitcher_stats"] = {}

    return matched


@st.cache_data(ttl=60 * 60)
def fetch_mlb_team_stats(season: int) -> pd.DataFrame:
    url = "https://statsapi.mlb.com/api/v1/standings"
    params = {"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"}
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
            run_diff = runs_scored - runs_allowed if runs_scored is not None and runs_allowed is not None else None

            profile = fetch_team_stat_profile(int(team_id), season) if team_id is not None else {}
            hitting = profile.get("hitting", {})
            pitching = profile.get("pitching", {})

            hitting_games = float(hitting.get("gamesPlayed") or 0)
            pitching_games = float(pitching.get("gamesPlayed") or 0)
            hitting_runs = float(hitting.get("runs") or 0)
            pitching_runs = float(pitching.get("runs") or 0)

            def parse_float(value):
                try:
                    return float(value) if value is not None else None
                except Exception:
                    return None

            rows.append(
                {
                    "team_id": team_id,
                    "team": team_name,
                    "wins": wins,
                    "losses": losses,
                    "win_pct": win_pct,
                    "run_diff": run_diff,
                    "hitting_ops": parse_float(hitting.get("ops")),
                    "hitting_runs_per_game": (hitting_runs / hitting_games) if hitting_games else None,
                    "pitching_era": parse_float(pitching.get("era")),
                    "pitching_whip": parse_float(pitching.get("whip")),
                    "pitching_runs_allowed_per_game": (pitching_runs / pitching_games) if pitching_games else None,
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset=["team"])
    if df.empty:
        raise ValueError("Could not load MLB standings data.")

    def zscore(series: pd.Series) -> pd.Series:
        clean = pd.to_numeric(series, errors="coerce")
        std = clean.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0, index=series.index)
        return (clean.fillna(clean.mean()) - clean.mean()) / std

    df["power_score"] = (
        zscore(df["win_pct"]) * 0.20
        + zscore(df["run_diff"]) * 0.20
        + zscore(df["hitting_ops"]) * 0.20
        + zscore(df["hitting_runs_per_game"]) * 0.10
        + zscore(-df["pitching_era"]) * 0.20
        + zscore(-df["pitching_whip"]) * 0.05
        + zscore(-df["pitching_runs_allowed_per_game"]) * 0.05
    )
    return df


@st.cache_data(ttl=60 * 30)
def fetch_nba_team_stats() -> pd.DataFrame:
    url = "https://www.espn.com/nba/standings"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    if len(tables) < 4:
        raise ValueError("Could not load NBA standings data.")

    def extract_team_name(raw: str) -> str:
        text = str(raw).strip()
        for team_name in NBA_TEAM_NAMES:
            if text.endswith(team_name):
                return team_name
        return text

    rows = []
    for names_table, stats_table, conference in [(tables[0], tables[1], "East"), (tables[2], tables[3], "West")]:
        team_names = [extract_team_name(value) for value in names_table.iloc[:, 0].tolist()]
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
            win_pct = as_float(row.get("PCT")) or (wins / games if games else 0.5)
            diff = as_float(row.get("DIFF"))
            ppg = as_float(row.get("PPG"))
            opp_ppg = as_float(row.get("OPP PPG"))
            l10 = str(row.get("L10", "")).strip()
            l10_pct = None
            if "-" in l10:
                try:
                    l10_w, l10_l = l10.split("-", 1)
                    total = float(l10_w) + float(l10_l)
                    l10_pct = float(l10_w) / total if total else None
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

    df["power_score"] = (
        zscore(df["win_pct"]) * 0.25
        + zscore(df["point_diff"]) * 0.25
        + zscore(df["ppg"]) * 0.20
        + zscore(-df["opp_ppg"]) * 0.20
        + zscore(df["l10_win_pct"]) * 0.10
    )
    return df


class OddsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_live_odds(self, sport_key: str, regions: str = "us") -> list[dict]:
        return fetch_live_odds(self.api_key, sport_key, regions)

    def fetch_scores(self, sport_key: str, days_from: int = 1) -> list[dict]:
        return fetch_scores(self.api_key, sport_key, days_from)

    def fetch_active_tennis_sport_keys(self) -> list[str]:
        return fetch_active_tennis_sport_keys(self.api_key)

    def fetch_active_tennis_odds(
        self,
        regions_priority: tuple[str, ...] = ("us", "uk", "eu", "au"),
        include_sport_keys: tuple[str, ...] = (),
    ) -> list[dict]:
        odds: list[dict] = []
        seen_event_ids: set[str] = set()

        active_keys = self.fetch_active_tennis_sport_keys()
        all_keys: list[str] = []
        for key in active_keys + list(include_sport_keys):
            if key and key not in all_keys:
                all_keys.append(key)

        for sport_key in all_keys:
            for regions in regions_priority:
                try:
                    rows = self.fetch_live_odds(sport_key, regions=regions)
                except Exception:
                    continue

                for row in rows:
                    event_id = str(row.get("id") or "")
                    if event_id and event_id in seen_event_ids:
                        continue
                    if event_id:
                        seen_event_ids.add(event_id)
                    odds.append(row)
        return odds


class StatsService:
    def fetch_mlb_team_stats(self, season: int) -> pd.DataFrame:
        return fetch_mlb_team_stats(season)

    def fetch_nba_team_stats(self) -> pd.DataFrame:
        return fetch_nba_team_stats()

    def fetch_mlb_game_context(self, odds_games: list[dict], season: int) -> dict:
        return fetch_mlb_game_context(odds_games, season)
