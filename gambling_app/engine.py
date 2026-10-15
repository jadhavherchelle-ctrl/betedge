from __future__ import annotations

import itertools
import re
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from .constants import MLB_TEAM_NAME_MAP, NBA_TEAM_NAMES


class BettingEngine:
    @staticmethod
    def american_to_decimal(american_odds: float) -> float:
        if american_odds > 0:
            return 1 + (american_odds / 100)
        return 1 + (100 / abs(american_odds))

    @staticmethod
    def american_to_implied_prob(american_odds: float) -> float:
        if american_odds > 0:
            return 100 / (american_odds + 100)
        return abs(american_odds) / (abs(american_odds) + 100)

    @staticmethod
    def normalize_mlb_team_name(name: str) -> str:
        return MLB_TEAM_NAME_MAP.get(name, name)

    @staticmethod
    def normalize_nba_team_name(name: str) -> str:
        mapping = {
            "LA Clippers": "Los Angeles Clippers",
            "L.A. Clippers": "Los Angeles Clippers",
            "Los Angeles Clippers": "Los Angeles Clippers",
        }
        return mapping.get(name, name)

    @staticmethod
    def get_best_h2h_prices(game: dict, normalizer) -> dict:
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
                    current = best[team]["price"]
                    if current is None or price > current:
                        best[team] = {"price": price, "book": book_name}
        return best

    @staticmethod
    def get_market_consensus_h2h(game: dict, normalizer) -> dict:
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
                        outcomes[team] = BettingEngine.american_to_implied_prob(price)
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
        return {home_team: home_prob, away_team: 1 - home_prob, "book_count": len(consensus_rows)}

    @staticmethod
    def logistic(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _pitcher_score(pitcher_stats: dict) -> float:
        if not pitcher_stats:
            return 0.0

        era = pitcher_stats.get("era")
        whip = pitcher_stats.get("whip")
        k_bb = pitcher_stats.get("k_bb")
        innings = pitcher_stats.get("innings_pitched")

        score = 0.0
        if era is not None:
            score += (4.20 - float(era)) * 0.12
        if whip is not None:
            score += (1.30 - float(whip)) * 0.22
        if k_bb is not None:
            score += (float(k_bb) - 2.50) * 0.05
        if innings is not None and float(innings) >= 20:
            score += 0.03
        return score

    @staticmethod
    def _bullpen_score(bullpen_ctx: dict) -> float:
        if not bullpen_ctx:
            return 0.0
        score = 0.0
        ip = bullpen_ctx.get("bullpen_ip_last3")
        era = bullpen_ctx.get("bullpen_era_last3")
        whip = bullpen_ctx.get("bullpen_whip_last3")

        if ip is not None:
            # More bullpen innings in last 3 days can indicate fatigue.
            score += (4.0 - float(ip)) * 0.025
        if era is not None:
            score += (4.20 - float(era)) * 0.03
        if whip is not None:
            score += (1.30 - float(whip)) * 0.06
        return score

    @staticmethod
    def _lineup_score(lineup_confirmed: bool, lineup_ops: float | None) -> float:
        if not lineup_confirmed or lineup_ops is None:
            return 0.0
        return (float(lineup_ops) - 0.720) * 0.45

    @staticmethod
    def _weather_score(weather_temp_f, weather_wind, weather_condition) -> float:
        score = 0.0

        def as_float(value):
            try:
                return float(value) if value is not None else None
            except Exception:
                return None

        temp = as_float(weather_temp_f)
        if temp is not None:
            if temp >= 85:
                score += 0.03
            elif temp >= 75:
                score += 0.015
            elif temp <= 45:
                score -= 0.02
            elif temp <= 35:
                score -= 0.03

        wind_text = str(weather_wind or "").lower()
        condition_text = str(weather_condition or "").lower()

        wind_match = re.search(r"(\d+(?:\.\d+)?)", wind_text)
        wind_mph = as_float(wind_match.group(1)) if wind_match else None
        if wind_mph is not None:
            wind_boost = 0.0
            if wind_mph >= 12:
                wind_boost = min(0.02 + ((wind_mph - 12) * 0.002), 0.05)
                if any(token in wind_text for token in ["out", "out to", "blowing out", "to rf", "to lf", "to cf"]):
                    score += wind_boost
                elif any(token in wind_text for token in ["in", "in from", "blowing in"]):
                    score -= wind_boost
                else:
                    score += wind_boost * 0.25

        if any(token in condition_text for token in ["rain", "drizzle", "storm", "thunder", "snow", "hail"]):
            score -= 0.03
        elif any(token in condition_text for token in ["clear", "sunny", "fair"]):
            score += 0.005
        elif any(token in condition_text for token in ["indoor", "dome"]):
            score += 0.002

        return float(np.clip(score, -0.06, 0.06))

    @staticmethod
    def build_game_edges(
        odds_games: list[dict],
        stats_df: pd.DataFrame,
        home_field_adv: float,
        normalizer,
        game_context: dict | None = None,
    ) -> pd.DataFrame:
        team_power = {normalizer(r["team"]): r["power_score"] for _, r in stats_df.iterrows()}
        rows = []

        for game in odds_games:
            home = normalizer(game.get("home_team"))
            away = normalizer(game.get("away_team"))
            if home not in team_power or away not in team_power:
                continue

            best_prices = BettingEngine.get_best_h2h_prices(game, normalizer)
            if best_prices.get(home, {}).get("price") is None or best_prices.get(away, {}).get("price") is None:
                continue

            home_price = best_prices[home]["price"]
            away_price = best_prices[away]["price"]

            market = BettingEngine.get_market_consensus_h2h(game, normalizer)
            market_home_prob = market.get(home, 0.5)
            market_away_prob = market.get(away, 0.5)

            context = (game_context or {}).get(f"{away} @ {home}", {})
            home_pitcher_stats = context.get("home_pitcher_stats", {})
            away_pitcher_stats = context.get("away_pitcher_stats", {})
            pitcher_diff = BettingEngine._pitcher_score(home_pitcher_stats) - BettingEngine._pitcher_score(away_pitcher_stats)

            home_bullpen = context.get("home_bullpen", {})
            away_bullpen = context.get("away_bullpen", {})
            bullpen_diff = BettingEngine._bullpen_score(home_bullpen) - BettingEngine._bullpen_score(away_bullpen)

            home_lineup_confirmed = bool(context.get("home_lineup_confirmed", False))
            away_lineup_confirmed = bool(context.get("away_lineup_confirmed", False))
            home_lineup_ops = context.get("home_lineup_ops")
            away_lineup_ops = context.get("away_lineup_ops")
            lineup_diff = BettingEngine._lineup_score(home_lineup_confirmed, home_lineup_ops) - BettingEngine._lineup_score(away_lineup_confirmed, away_lineup_ops)

            weather_score = BettingEngine._weather_score(
                context.get("weather_temp_f"),
                context.get("weather_wind"),
                context.get("weather_condition"),
            )

            power_diff = team_power[home] - team_power[away] + home_field_adv + pitcher_diff + bullpen_diff + lineup_diff
            # Weather should only make a small difference. Favorable weather slightly amplifies the current edge.
            # Poor weather slightly dampens confidence.
            power_diff += weather_score * (1 if power_diff >= 0 else -1)
            stat_model_prob = float(BettingEngine.logistic(power_diff))
            home_model_prob = float(np.clip((stat_model_prob * 0.72) + (market_home_prob * 0.28), 0.01, 0.99))
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
                        "event_id": game.get("id"),
                        "sport_key": game.get("sport_key"),
                        "start_utc": start_time,
                        "team": home,
                        "opponent": away,
                        "side": "home",
                        "book": best_prices[home]["book"],
                        "books": market.get("book_count", 0),
                        "home_pitcher": context.get("home_pitcher_name"),
                        "away_pitcher": context.get("away_pitcher_name"),
                        "home_pitcher_era": home_pitcher_stats.get("era"),
                        "away_pitcher_era": away_pitcher_stats.get("era"),
                        "home_bullpen_ip_last3": home_bullpen.get("bullpen_ip_last3"),
                        "away_bullpen_ip_last3": away_bullpen.get("bullpen_ip_last3"),
                        "home_lineup_confirmed": home_lineup_confirmed,
                        "away_lineup_confirmed": away_lineup_confirmed,
                        "home_lineup_ops": home_lineup_ops,
                        "away_lineup_ops": away_lineup_ops,
                        "weather_temp_f": context.get("weather_temp_f"),
                        "weather_wind": context.get("weather_wind"),
                        "weather_condition": context.get("weather_condition"),
                        "american_odds": home_price,
                        "decimal_odds": BettingEngine.american_to_decimal(home_price),
                        "implied_prob_no_vig": market_home_prob,
                        "model_prob": home_model_prob,
                        "edge": home_model_prob - market_home_prob,
                    },
                    {
                        "game": f"{away} @ {home}",
                        "event_id": game.get("id"),
                        "sport_key": game.get("sport_key"),
                        "start_utc": start_time,
                        "team": away,
                        "opponent": home,
                        "side": "away",
                        "book": best_prices[away]["book"],
                        "books": market.get("book_count", 0),
                        "home_pitcher": context.get("home_pitcher_name"),
                        "away_pitcher": context.get("away_pitcher_name"),
                        "home_pitcher_era": home_pitcher_stats.get("era"),
                        "away_pitcher_era": away_pitcher_stats.get("era"),
                        "home_bullpen_ip_last3": home_bullpen.get("bullpen_ip_last3"),
                        "away_bullpen_ip_last3": away_bullpen.get("bullpen_ip_last3"),
                        "home_lineup_confirmed": home_lineup_confirmed,
                        "away_lineup_confirmed": away_lineup_confirmed,
                        "home_lineup_ops": home_lineup_ops,
                        "away_lineup_ops": away_lineup_ops,
                        "weather_temp_f": context.get("weather_temp_f"),
                        "weather_wind": context.get("weather_wind"),
                        "weather_condition": context.get("weather_condition"),
                        "american_odds": away_price,
                        "decimal_odds": BettingEngine.american_to_decimal(away_price),
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

    @staticmethod
    def build_parlays(plays_df: pd.DataFrame, max_legs: int, top_n: int) -> pd.DataFrame:
        sort_cols = [col for col in ["edge", "ev_per_1u"] if col in plays_df.columns]
        if not sort_cols:
            return pd.DataFrame()

        ranked = plays_df.sort_values(by=sort_cols, ascending=False).copy()
        seed_games = ranked.head(top_n)["game"].drop_duplicates().tolist()
        picks = ranked[ranked["game"].isin(seed_games)].copy()
        if picks.empty:
            return pd.DataFrame()

        parlays = []
        for legs in range(2, max_legs + 1):
            for combo in itertools.combinations(picks.to_dict("records"), legs):
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
                        "event_ids": tuple(c.get("event_id") for c in combo),
                        "sport_keys": tuple(c.get("sport_key") for c in combo),
                        "teams": tuple(c["team"] for c in combo),
                        "sides": tuple(c.get("side") for c in combo),
                        "leg_odds": tuple(c["american_odds"] for c in combo),
                        "favorite_legs": sum(1 for c in combo if float(c["american_odds"]) < 0),
                        "underdog_legs": sum(1 for c in combo if float(c["american_odds"]) > 0),
                        "decimal_odds": dec_odds,
                        "implied_prob": implied_prob,
                        "model_prob": model_prob,
                        "edge": edge,
                        "ev_per_1u": ev,
                    }
                )

        df = pd.DataFrame(parlays)
        if not df.empty:
            df["contrarian_score"] = df["underdog_legs"] / df["legs"]
            df["parlay_style"] = np.where(
                df["contrarian_score"] >= 0.5,
                "Contrarian",
                np.where(df["contrarian_score"] <= 0.5, "Favorite-lean", "Mixed"),
            )
            df = df.sort_values(by="ev_per_1u", ascending=False)
        return df

    @staticmethod
    def _parlay_overlap_score(candidate: pd.Series, selected: list[pd.Series]) -> int:
        candidate_games = set(candidate["games"])
        candidate_teams = set(candidate["teams"])
        overlap = 0
        for parlay in selected:
            overlap += len(candidate_games.intersection(set(parlay["games"])))
            overlap += len(candidate_teams.intersection(set(parlay["teams"])))
        return overlap

    @staticmethod
    def select_diversified_parlays(
        parlay_df: pd.DataFrame,
        max_parlays: int,
        diversification_bias: float,
        max_team_uses: int,
        contrarian_mix_pct: float,
    ) -> pd.DataFrame:
        if parlay_df.empty:
            return pd.DataFrame()

        scored = parlay_df.copy()
        # Allow slightly-negative EV parlays into the pool so there are enough
        # candidates to fill the full parlay count and deploy the whole bankroll.
        scored = scored[scored["ev_per_1u"] >= -0.05].copy()
        if scored.empty:
            return pd.DataFrame()

        scored["unique_teams"] = scored["teams"].apply(lambda x: len(set(x)))
        if "contrarian_score" in scored.columns:
            scored["contrarian_score"] = pd.to_numeric(scored["contrarian_score"], errors="coerce").fillna(0.0)
        else:
            scored["contrarian_score"] = 0.0
        contrarian_mix_pct = float(np.clip(contrarian_mix_pct, 0.0, 1.0))
        scored["base_score"] = scored["ev_per_1u"] * (1 + (scored["unique_teams"] * 0.06)) * (1 + (contrarian_mix_pct * scored["contrarian_score"] * 0.35))
        scored = scored.sort_values(by="base_score", ascending=False)

        contrarian_target = int(round(max_parlays * contrarian_mix_pct))
        contrarian_target = max(0, min(max_parlays, contrarian_target))
        opposing_side_pressure = float(np.clip(0.25 + (contrarian_mix_pct * 1.25), 0.0, 1.0))

        selected: list[pd.Series] = []
        team_use_counts: dict[str, int] = {}
        game_side_counts: dict[str, dict[str, int]] = {}
        selected_keys: set[tuple] = set()

        def game_side_balance_bonus(row: pd.Series) -> float:
            bonus = 0.0
            games = list(row.get("games", []))
            sides = list(row.get("sides", []))
            for idx, game in enumerate(games):
                side = str(sides[idx]) if idx < len(sides) and sides[idx] is not None else ""
                if not game or not side:
                    continue
                counts = game_side_counts.get(game, {})
                same = int(counts.get(side, 0))
                opposite = int(sum(v for k, v in counts.items() if k != side))
                if opposite > 0 and same == 0:
                    # Strongly reward adding the opposite side of a matchup already present.
                    bonus += 0.26 + (0.24 * opposing_side_pressure)
                elif opposite > same:
                    bonus += min(0.10 * (opposite - same), 0.28)
                elif same > 0 and opposite == 0:
                    # Penalize repeating only one side of the same game across portfolio.
                    bonus -= min((0.12 + (0.18 * opposing_side_pressure)) * same, 0.36)
                elif same > 0:
                    bonus -= min(0.06 * same, 0.18)
            return bonus

        def pick_from_pool(pool: pd.DataFrame, max_needed: int) -> None:
            nonlocal selected, team_use_counts
            while len(selected) < max_parlays and max_needed > 0:
                best_idx = None
                best_row = None
                best_adjusted = None
                best_overlap = 0
                best_side_bonus = 0.0

                for idx, row in pool.iterrows():
                    key = (row.get("parlay"), row.get("games"), row.get("teams"))
                    if key in selected_keys:
                        continue

                    teams = list(row["teams"])
                    if any(team_use_counts.get(team, 0) >= max_team_uses for team in teams):
                        continue

                    overlap = BettingEngine._parlay_overlap_score(row, selected)
                    side_bonus = game_side_balance_bonus(row)
                    adjusted_score = (float(row["base_score"]) * (1 + side_bonus)) / (1 + (diversification_bias * overlap))
                    if adjusted_score <= 0:
                        continue

                    if best_adjusted is None or adjusted_score > best_adjusted:
                        best_adjusted = adjusted_score
                        best_idx = idx
                        best_row = row
                        best_overlap = overlap
                        best_side_bonus = side_bonus

                if best_row is None or best_idx is None or best_adjusted is None:
                    break

                row = best_row.copy()
                teams = list(row["teams"])
                row["adjusted_score"] = float(best_adjusted)
                row["overlap_score"] = int(best_overlap)
                row["side_balance_bonus"] = float(best_side_bonus)
                selected.append(row)
                selected_keys.add((row.get("parlay"), row.get("games"), row.get("teams")))

                for team in teams:
                    team_use_counts[team] = team_use_counts.get(team, 0) + 1

                row_games = list(row.get("games", []))
                row_sides = list(row.get("sides", []))
                for idx, game in enumerate(row_games):
                    side = str(row_sides[idx]) if idx < len(row_sides) and row_sides[idx] is not None else ""
                    if not game or not side:
                        continue
                    game_counts = game_side_counts.setdefault(game, {})
                    game_counts[side] = game_counts.get(side, 0) + 1

                max_needed -= 1

        if contrarian_target > 0:
            contrarian_pool = scored[scored["contrarian_score"] >= 0.5].copy()
            if not contrarian_pool.empty:
                pick_from_pool(contrarian_pool, contrarian_target)

        if len(selected) < max_parlays:
            remaining_pool = scored.copy()
            pick_from_pool(remaining_pool, max_parlays - len(selected))

        # Enforce hedge coverage for games that appear multiple times on only one side.
        # This supports "failsafe" behavior (e.g., include some Red Sox exposure when
        # Yankees are used in several parlays from the same matchup).
        def side_for_game(row: pd.Series, game_name: str) -> str:
            games = list(row.get("games", []))
            sides = list(row.get("sides", []))
            for idx, gm in enumerate(games):
                if str(gm) == str(game_name):
                    return str(sides[idx]) if idx < len(sides) and sides[idx] is not None else ""
            return ""

        def selected_game_side_counts(rows: list[pd.Series]) -> dict[str, dict[str, int]]:
            out: dict[str, dict[str, int]] = {}
            for row in rows:
                games = list(row.get("games", []))
                sides = list(row.get("sides", []))
                for idx, gm in enumerate(games):
                    side = str(sides[idx]) if idx < len(sides) and sides[idx] is not None else ""
                    if not gm or not side:
                        continue
                    counts = out.setdefault(str(gm), {})
                    counts[side] = counts.get(side, 0) + 1
            return out

        def selected_team_counts(rows: list[pd.Series]) -> dict[str, int]:
            out: dict[str, int] = {}
            for row in rows:
                for tm in list(row.get("teams", [])):
                    out[tm] = out.get(tm, 0) + 1
            return out

        for _ in range(10):
            if not selected:
                break

            game_counts = selected_game_side_counts(selected)
            # Prioritize games with the highest single-side concentration.
            imbalanced = [
                (gm, counts)
                for gm, counts in game_counts.items()
                if len(counts) == 1 and sum(counts.values()) >= 2
            ]
            if not imbalanced:
                break

            imbalanced.sort(key=lambda item: sum(item[1].values()), reverse=True)
            changed = False

            for game_name, counts in imbalanced:
                dominant_side = next(iter(counts.keys()))

                # Best opposite-side candidate not already selected.
                # Use full parlay universe here (not only positive-EV rows) so hedge legs
                # can still appear when they are slightly negative EV but improve failsafe coverage.
                best_cand = None
                best_cand_score = None
                for _, cand in parlay_df.iterrows():
                    cand_key = (cand.get("parlay"), cand.get("games"), cand.get("teams"))
                    if cand_key in selected_keys:
                        continue

                    cand_side = side_for_game(cand, game_name)
                    if not cand_side or cand_side == dominant_side:
                        continue

                    overlap = BettingEngine._parlay_overlap_score(cand, selected)
                    side_bonus = game_side_balance_bonus(cand)
                    ev = float(cand.get("ev_per_1u", 0.0))
                    model_prob = float(cand.get("model_prob", 0.0))
                    # Hedge score: allow modestly negative EV if the opposite-side hedge is strong.
                    # Very negative EV parlays still get filtered out.
                    cand_score = (
                        (0.70 * max(ev, -0.20))
                        + (0.30 * model_prob)
                        + 0.40
                        + (0.10 * side_bonus)
                    ) / (1 + (diversification_bias * overlap))
                    if cand_score <= 0:
                        continue

                    if best_cand_score is None or cand_score > best_cand_score:
                        best_cand = cand
                        best_cand_score = cand_score

                if best_cand is None or best_cand_score is None:
                    continue

                # Replace the weakest selected parlay using the dominant side of this same game.
                removable_idx = []
                for idx, row in enumerate(selected):
                    if side_for_game(row, game_name) == dominant_side:
                        removable_idx.append(idx)

                removable_idx.sort(key=lambda idx: float(selected[idx].get("adjusted_score", 0.0)))
                if not removable_idx:
                    continue

                team_counts_now = selected_team_counts(selected)
                cand_teams = list(best_cand.get("teams", []))

                replaced = False
                for ridx in removable_idx:
                    to_remove = selected[ridx]
                    remove_teams = list(to_remove.get("teams", []))

                    feasible = True
                    for tm in cand_teams:
                        projected = team_counts_now.get(tm, 0) - (1 if tm in remove_teams else 0) + 1
                        if projected > max_team_uses:
                            feasible = False
                            break
                    if not feasible:
                        continue

                    remove_key = (to_remove.get("parlay"), to_remove.get("games"), to_remove.get("teams"))
                    selected_keys.discard(remove_key)
                    selected[ridx] = best_cand.copy()
                    selected[ridx]["adjusted_score"] = float(best_cand_score)
                    selected[ridx]["overlap_score"] = int(BettingEngine._parlay_overlap_score(best_cand, selected))
                    selected[ridx]["side_balance_bonus"] = float(game_side_balance_bonus(best_cand))
                    selected_keys.add((best_cand.get("parlay"), best_cand.get("games"), best_cand.get("teams")))
                    replaced = True
                    changed = True
                    break

                if changed and replaced:
                    break

            if not changed:
                break

        # ── Heavy-favourite protection ─────────────────────────────────────────
        # If any team with very negative american odds (big favourite) appears
        # in 2+ selected parlays all on the same side, insert up to 2 extra
        # parlays that include the opposing team.  This way a single upset
        # doesn't wipe the whole portfolio.
        HEAVY_FAV_ODDS      = -140   # more negative → "heavy favourite"
        HEAVY_FAV_MIN_COUNT = 2      # trigger when fav appears this many times

        def _min_leg_odds_for_side(rows: list[pd.Series], game_name: str, side_match: str) -> float | None:
            best: float | None = None
            for row in rows:
                games_ = list(row.get("games", []))
                sides_ = list(row.get("sides", []))
                lodds_ = list(row.get("leg_odds", []))
                for i, gm in enumerate(games_):
                    if str(gm) != str(game_name):
                        continue
                    if (str(sides_[i]) if i < len(sides_) else "") != side_match:
                        continue
                    if i < len(lodds_) and lodds_[i] is not None:
                        v = float(lodds_[i])
                        if best is None or v < best:
                            best = v
            return best

        hedge_added = 0
        gsc_final = selected_game_side_counts(selected)
        for game_name, counts in gsc_final.items():
            if hedge_added >= 2:
                break
            for dominant_side, count_ in counts.items():
                if count_ < HEAVY_FAV_MIN_COUNT:
                    continue
                min_odds = _min_leg_odds_for_side(selected, game_name, dominant_side)
                if min_odds is None or min_odds >= HEAVY_FAV_ODDS:
                    continue
                opp_side = "away" if dominant_side == "home" else "home"
                if gsc_final.get(game_name, {}).get(opp_side, 0) >= 1:
                    continue  # already hedged on the other side

                best_hcand: pd.Series | None = None
                best_hcand_score: float | None = None
                for _, cand in parlay_df.iterrows():
                    ckey = (cand.get("parlay"), cand.get("games"), cand.get("teams"))
                    if ckey in selected_keys:
                        continue
                    if side_for_game(cand, game_name) != opp_side:
                        continue
                    cand_teams_ = list(cand.get("teams", []))
                    if any(team_use_counts.get(t, 0) >= max_team_uses for t in cand_teams_):
                        continue
                    ev_   = float(cand.get("ev_per_1u", 0.0))
                    mp_   = float(cand.get("model_prob", 0.0))
                    ov_   = BettingEngine._parlay_overlap_score(cand, selected)
                    # Generous scoring: tolerate negative EV for the protection hedge.
                    hs = (0.50 * max(ev_, -0.40) + 0.50 * mp_ + 0.60) / (1 + 0.20 * ov_)
                    if best_hcand_score is None or hs > best_hcand_score:
                        best_hcand = cand
                        best_hcand_score = hs

                if best_hcand is None or (best_hcand_score or 0) <= 0:
                    continue

                row_h = best_hcand.copy()
                row_h["adjusted_score"] = float(best_hcand_score)
                row_h["overlap_score"] = int(BettingEngine._parlay_overlap_score(best_hcand, selected))
                row_h["side_balance_bonus"] = 0.5
                selected.append(row_h)
                hkey = (best_hcand.get("parlay"), best_hcand.get("games"), best_hcand.get("teams"))
                selected_keys.add(hkey)
                for t in list(best_hcand.get("teams", [])):
                    team_use_counts[t] = team_use_counts.get(t, 0) + 1
                for idx_h, gm_h in enumerate(list(best_hcand.get("games", []))):
                    sl_h = list(best_hcand.get("sides", []))
                    s_h  = str(sl_h[idx_h]) if idx_h < len(sl_h) else ""
                    if gm_h and s_h:
                        game_side_counts.setdefault(str(gm_h), {})[s_h] = game_side_counts.get(str(gm_h), {}).get(s_h, 0) + 1
                gsc_final = selected_game_side_counts(selected)
                hedge_added += 1
                break

        if not selected:
            return pd.DataFrame()

        selected_df = pd.DataFrame(selected)
        return selected_df.sort_values(by="adjusted_score", ascending=False)

    @staticmethod
    def allocate_bankroll_to_parlays(
        parlay_df: pd.DataFrame,
        bankroll: float,
        max_parlays: int,
        max_bet_pct: float,
        fractional_kelly: float,
        diversification_bias: float,
        max_team_uses: int,
        max_team_exposure_pct: float,
        contrarian_mix_pct: float,
    ) -> pd.DataFrame:
        if parlay_df.empty or bankroll <= 0:
            return pd.DataFrame()

        candidates = BettingEngine.select_diversified_parlays(
            parlay_df,
            max_parlays=max_parlays,
            diversification_bias=diversification_bias,
            max_team_uses=max_team_uses,
            contrarian_mix_pct=contrarian_mix_pct,
        )
        if candidates.empty:
            return pd.DataFrame()

        weights = candidates["ev_per_1u"].clip(lower=0).copy()
        weights = weights * (1 + (candidates["unique_teams"] * 0.03))
        shape_exponent = 0.45 + (0.55 * max(min(fractional_kelly, 1.0), 0.0))
        weights = weights.pow(shape_exponent)

        if float(weights.sum()) <= 0:
            return pd.DataFrame()

        kelly_fraction = weights / float(weights.sum())

        # Water-fill: iteratively normalize → clip → repeat until the full
        # bankroll is deployed.  Parlays that hit the per-parlay cap give their
        # excess weight to the remaining parlays, so every dollar is allocated.
        for _ in range(40):
            total = float(kelly_fraction.sum())
            if total <= 0:
                return pd.DataFrame()
            kelly_fraction = kelly_fraction / total          # normalize to 1.0
            clipped = kelly_fraction.clip(lower=0, upper=max_bet_pct)
            if float((clipped - kelly_fraction).abs().sum()) < 1e-9:
                break                                        # converged, sum == 1
            kelly_fraction = clipped

        kelly_fraction = kelly_fraction.clip(lower=0, upper=max_bet_pct)
        if float(kelly_fraction.sum()) <= 0:
            return pd.DataFrame()

        out = candidates.copy()
        out["kelly_fraction"] = kelly_fraction
        out["stake"] = bankroll * kelly_fraction
        out = out[out["stake"] > 0].copy()
        if out.empty:
            return pd.DataFrame()

        max_team_exposure_pct = float(np.clip(max_team_exposure_pct, 0.01, 1.0))
        team_cap_dollars = float(bankroll) * max_team_exposure_pct

        def team_exposure(df: pd.DataFrame) -> pd.Series:
            return (
                df.assign(team_list=df["teams"]) 
                .explode("team_list")
                .groupby("team_list")["stake"]
                .sum()
            )

        # Iteratively scale down stakes on parlays that violate team concentration cap.
        for _ in range(8):
            exposures = team_exposure(out)
            violating = exposures[exposures > team_cap_dollars]
            if violating.empty:
                break

            scale = pd.Series(1.0, index=out.index, dtype=float)
            for team_name, exposure_amount in violating.items():
                ratio = float(team_cap_dollars / exposure_amount)
                impacted = out["teams"].apply(lambda teams: team_name in teams)
                scale.loc[impacted] = np.minimum(scale.loc[impacted], ratio)

            out["stake"] = out["stake"] * scale
            out = out[out["stake"] > 0.01].copy()
            if out.empty:
                return pd.DataFrame()

        # Always deploy the full bankroll: proportionally scale up remaining stakes
        # after concentration controls reduced some of them.
        total_deployed = float(out["stake"].sum())
        if total_deployed > 0 and total_deployed < bankroll * 0.995:
            out["stake"] = out["stake"] * (bankroll / total_deployed)

        # Recompute kelly fraction after concentration controls.
        out["kelly_fraction"] = out["stake"] / float(bankroll)

        out["potential_payout"] = out["stake"] * out["decimal_odds"]
        out["potential_profit"] = out["potential_payout"] - out["stake"]
        out["expected_profit"] = out["stake"] * out["ev_per_1u"]
        out["win_loss_pct"] = out["model_prob"] * 100
        out["loss_pct"] = (1 - out["model_prob"]) * 100
        out["team_count"] = out["teams"].apply(lambda teams: len(set(teams)))
        return out.sort_values(by="expected_profit", ascending=False)

    @staticmethod
    def summarize_allocation_risk(alloc_df: pd.DataFrame) -> dict:
        if alloc_df.empty:
            return {"lose_all_prob": 0.0, "at_least_one_win_prob": 0.0, "total_stake": 0.0, "total_payout_if_all_win": 0.0, "total_profit_if_all_win": 0.0, "expected_profit": 0.0, "max_team_exposure": 0.0}

        win_probs = alloc_df["model_prob"].clip(lower=0, upper=1)
        lose_all_prob = float(np.prod(1 - win_probs))
        team_exposure = (
            alloc_df.assign(team_list=alloc_df["teams"]) 
            .explode("team_list")
            .groupby("team_list")["stake"]
            .sum()
        )
        max_team_exposure = float(team_exposure.max()) if not team_exposure.empty else 0.0
        return {
            "lose_all_prob": lose_all_prob,
            "at_least_one_win_prob": 1 - lose_all_prob,
            "total_stake": float(alloc_df["stake"].sum()),
            "total_payout_if_all_win": float(alloc_df["potential_payout"].sum()),
            "total_profit_if_all_win": float(alloc_df["potential_profit"].sum()),
            "expected_profit": float(alloc_df["expected_profit"].sum()),
            "max_team_exposure": max_team_exposure,
        }

    @staticmethod
    def summarize_matchup_balance(alloc_df: pd.DataFrame) -> dict:
        if alloc_df.empty or "games" not in alloc_df.columns or "sides" not in alloc_df.columns:
            return {"balanced_game_rate": 0.0, "game_count": 0}

        game_side_map: dict[str, set[str]] = {}
        for _, row in alloc_df.iterrows():
            games = list(row.get("games", []))
            sides = list(row.get("sides", []))
            for idx, game in enumerate(games):
                side = str(sides[idx]) if idx < len(sides) and sides[idx] is not None else ""
                if not game or not side:
                    continue
                game_side_map.setdefault(str(game), set()).add(side)

        if not game_side_map:
            return {"balanced_game_rate": 0.0, "game_count": 0}

        balanced_games = sum(1 for side_set in game_side_map.values() if len(side_set) >= 2)
        return {
            "balanced_game_rate": balanced_games / len(game_side_map),
            "game_count": len(game_side_map),
        }

    @staticmethod
    def filter_later_today(edges_df: pd.DataFrame) -> pd.DataFrame:
        if edges_df.empty or "start_utc" not in edges_df.columns:
            return edges_df

        out = edges_df.copy()
        out["start_utc"] = pd.to_datetime(out["start_utc"], utc=True, errors="coerce")

        local_now = datetime.now().astimezone()
        if local_now.tzinfo is None:
            return out

        end_of_day_local = local_now.replace(hour=23, minute=59, second=59, microsecond=999999)
        mask = (out["start_utc"].notna()) & (out["start_utc"] >= local_now.astimezone(timezone.utc)) & (out["start_utc"] <= end_of_day_local.astimezone(timezone.utc))
        return out[mask].copy()

    @staticmethod
    def filter_by_local_date(edges_df: pd.DataFrame, target_local_date: date) -> pd.DataFrame:
        if edges_df.empty or "start_utc" not in edges_df.columns:
            return edges_df

        out = edges_df.copy()
        out["start_utc"] = pd.to_datetime(out["start_utc"], utc=True, errors="coerce")

        local_tz = datetime.now().astimezone().tzinfo
        if local_tz is None:
            return out

        local_dates = out["start_utc"].dt.tz_convert(local_tz).dt.date
        return out[(out["start_utc"].notna()) & (local_dates == target_local_date)].copy()
