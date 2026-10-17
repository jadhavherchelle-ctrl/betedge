from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import requests
import streamlit as st

from .backtest_engine import run_backtest, HistoricalOddsUnavailable
from .controller import BettingController
from .engine import BettingEngine
from .services import OddsAPIClient, StatsService
from .sports.tennis import normalize_tennis_player_name

_EDGE_HIGH = 5.0
_EDGE_MED = 2.0


class BettingApp:
    def __init__(self) -> None:
        self.stats_service = StatsService()
        self.engine = BettingEngine()
        self.controller = BettingController(self.stats_service, self.engine)
        self.saved_parlays_path = Path(__file__).resolve().parent.parent / "saved_parlays.json"
        self.simulation_tracker_path = Path(__file__).resolve().parent.parent / "simulation_tracker.json"

    @staticmethod
    def _normalize_tennis_player_name(name: str | None) -> str:
        return normalize_tennis_player_name(name)

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_saved_parlays(self) -> list[dict]:
        if not self.saved_parlays_path.exists():
            return []
        try:
            with self.saved_parlays_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, list):
                return [r for r in payload if isinstance(r, dict)]
        except Exception:
            pass
        return []

    def _save_parlays(self, parlays: list[dict]) -> None:
        merged = self._load_saved_parlays() + parlays
        with self.saved_parlays_path.open("w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2, default=str)

    def _load_simulation_tickets(self) -> list[dict]:
        if not self.simulation_tracker_path.exists():
            return []
        try:
            with self.simulation_tracker_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, list):
                return [r for r in payload if isinstance(r, dict)]
        except Exception:
            pass
        return []

    def _save_simulation_tickets(self, tickets: list[dict]) -> None:
        with self.simulation_tracker_path.open("w", encoding="utf-8") as fh:
            json.dump(tickets, fh, indent=2, default=str)

    def _append_simulation_tickets(self, tickets: list[dict]) -> int:
        if not tickets:
            return 0
        self._save_simulation_tickets(self._load_simulation_tickets() + tickets)
        return len(tickets)

    def _normalize_for_sport(self, name: str | None, sport_key: str | None) -> str:
        text = "" if name is None else str(name).strip()
        sk = str(sport_key or "")
        if sk.startswith("baseball_"):
            return BettingEngine.normalize_mlb_team_name(text)
        if sk.startswith("basketball_"):
            return BettingEngine.normalize_nba_team_name(text)
        return self._normalize_tennis_player_name(text)

    # ── grading ──────────────────────────────────────────────────────────────

    def _grade_simulation_tickets(self, client: OddsAPIClient) -> tuple[int, int]:
        tickets = self._load_simulation_tickets()
        if not tickets:
            return (0, 0)

        score_cache: dict[str, dict[str, str]] = {}

        def winner_map_for_sport(sport_key: str) -> dict[str, str]:
            if sport_key in score_cache:
                return score_cache[sport_key]
            winners: dict[str, str] = {}
            try:
                scores = client.fetch_scores(sport_key, days_from=3)
            except Exception:
                score_cache[sport_key] = winners
                return winners
            for event in scores:
                if not event.get("completed"):
                    continue
                event_id = event.get("id")
                score_rows = event.get("scores") or []
                if not event_id or len(score_rows) < 2:
                    continue
                parsed = []
                for row in score_rows:
                    try:
                        parsed.append((row.get("name"), float(row.get("score"))))
                    except Exception:
                        continue
                if len(parsed) < 2:
                    continue
                parsed.sort(key=lambda x: x[1], reverse=True)
                if parsed[0][1] == parsed[1][1]:
                    continue
                winners[str(event_id)] = self._normalize_for_sport(parsed[0][0], sport_key)
            score_cache[sport_key] = winners
            return winners

        settled = updated = 0
        for ticket in tickets:
            if ticket.get("status") == "settled":
                continue
            teams = list(ticket.get("teams", []))
            event_ids = [str(x) for x in ticket.get("event_ids", [])]
            sport_keys = list(ticket.get("sport_keys", []))
            if not teams or not event_ids or len(teams) != len(event_ids):
                continue

            games = list(ticket.get("games", []))
            leg_results: list[dict] = []
            unresolved = 0

            for idx, event_id in enumerate(event_ids):
                sport_key = str(sport_keys[idx]) if idx < len(sport_keys) and sport_keys[idx] else ""
                if not sport_key:
                    unresolved += 1
                    continue
                winner = winner_map_for_sport(sport_key).get(event_id)
                if winner is None:
                    unresolved += 1
                    continue
                pick = self._normalize_for_sport(teams[idx], sport_key)
                leg_results.append({
                    "idx": idx + 1,
                    "game": games[idx] if idx < len(games) else "",
                    "pick": pick,
                    "winner": winner,
                    "won": pick == winner,
                })

            stake = float(ticket.get("stake", 0.0) or 0.0)
            decimal_odds = float(ticket.get("decimal_odds", 0.0) or 0.0)
            failed = [r for r in leg_results if not r["won"]]
            if failed:
                won = False
            elif unresolved > 0:
                continue
            else:
                won = True

            profit_u = stake * (decimal_odds - 1.0) if won else -stake
            ticket["status"] = "settled"
            ticket["result"] = "won" if won else "lost"
            ticket["profit_u"] = float(round(profit_u, 4))
            ticket["failed_legs"] = len(failed)
            ticket["failed_legs_detail"] = (
                " | ".join(f"Leg {r['idx']}: {r['pick']} lost (winner: {r['winner']})" for r in failed)
                if failed else ""
            )
            ticket["settled_at"] = datetime.now().isoformat(timespec="seconds")
            settled += 1
            updated += 1

        if updated:
            self._save_simulation_tickets(tickets)
        return settled, updated

    # ── optimizer helpers ─────────────────────────────────────────────────────

    def _score_settings_for_edges(
        self,
        edges_df: pd.DataFrame,
        settings: dict,
        bankroll: float,
        only_later_today: bool,
        selected_local_date: date | None,
    ) -> dict:
        empty = {"expected_profit": 0.0, "at_least_one_win_prob": 0.0, "max_team_exposure_pct": 0.0, "balanced_game_rate": 0.0}
        if edges_df.empty:
            return empty

        working = edges_df.copy()
        if selected_local_date is not None:
            working = BettingEngine.filter_by_local_date(working, selected_local_date)
        if only_later_today and (selected_local_date is None or selected_local_date == datetime.now().astimezone().date()):
            working = BettingEngine.filter_later_today(working)
        if working.empty:
            return empty

        filtered = BettingEngine.filter_realistic(
            working, float(settings["min_edge"]), float(st.session_state.get("min_win_prob", 50.0))
        )
        if filtered.empty:
            return empty

        seed_games = (
            filtered.sort_values(by="pick_score", ascending=False)
            .head(int(settings["parlay_top_n"]))["game"]
            .drop_duplicates().tolist()
        )
        parlay_input = working[working["game"].isin(seed_games)].copy()
        if parlay_input.empty:
            return empty

        parlay_df = self.engine.build_parlays(parlay_input, max_legs=int(settings["parlay_legs"]), top_n=int(settings["parlay_top_n"]))
        if parlay_df.empty:
            return empty
        if len(parlay_df) > 2500:
            parlay_df = parlay_df.sort_values(by=["ev_per_1u", "edge"], ascending=False).head(2500).copy()

        alloc_df = self.engine.allocate_bankroll_to_parlays(
            parlay_df=parlay_df,
            bankroll=float(bankroll),
            max_parlays=int(settings["max_alloc_parlays"]),
            max_bet_pct=float(settings["max_bet_pct"]) / 100,
            fractional_kelly=float(settings["fractional_kelly"]),
            diversification_bias=float(settings["diversification_bias"]),
            max_team_uses=int(settings["max_team_uses"]),
            max_team_exposure_pct=float(settings["max_team_exposure_pct"]) / 100,
            contrarian_mix_pct=float(settings["contrarian_mix_pct"]) / 100,
        )
        if alloc_df.empty:
            return empty

        risk = self.engine.summarize_allocation_risk(alloc_df)
        balance = self.engine.summarize_matchup_balance(alloc_df)
        return {
            "expected_profit": float(risk["expected_profit"]),
            "at_least_one_win_prob": float(risk["at_least_one_win_prob"]),
            "max_team_exposure_pct": float(risk["max_team_exposure"]) / bankroll if bankroll > 0 else 0.0,
            "balanced_game_rate": float(balance["balanced_game_rate"]),
        }

    def _optimize_settings(
        self,
        baseball_odds, baseball_stats, baseball_context,
        basketball_odds, basketball_stats,
        tennis_odds, tennis_stats,
        bankroll, only_later_today, selected_local_date, objective,
    ) -> dict:
        search_space = {
            "home_field_adv": [-0.05, 0.00, 0.03, 0.06, 0.09, 0.12],
            "min_edge": [0.0, 1.0, 2.0, 3.0, 5.0],
            "parlay_legs": [2, 3, 4],
            "parlay_top_n": [6, 8, 10, 12, 14],
            "max_alloc_parlays": [6, 8, 10, 12, 15],
            "fractional_kelly": [0.20, 0.35, 0.50, 0.65],
            "diversification_bias": [0.4, 0.8, 1.2, 1.6],
            "max_team_uses": [1, 2, 3],
            "max_bet_pct": [8.0, 12.0, 15.0, 20.0, 25.0],
            "max_team_exposure_pct": [20.0, 25.0, 30.0, 35.0, 40.0],
            "contrarian_mix_pct": [0.0, 10.0, 20.0, 30.0, 40.0],
        }

        best = {k: float(st.session_state.get(k, v[0])) if isinstance(v[0], float) else int(st.session_state.get(k, v[0]))
                for k, v in search_space.items()}

        edges_cache: dict[float, tuple] = {}
        search_started = perf_counter()
        eval_count = 0

        def get_edges(home_adv: float) -> tuple:
            key = round(float(home_adv), 4)
            if key not in edges_cache:
                edges_cache[key] = (
                    self.engine.build_game_edges(baseball_odds, baseball_stats, float(home_adv), BettingEngine.normalize_mlb_team_name, game_context=baseball_context),
                    self.engine.build_game_edges(basketball_odds, basketball_stats, float(home_adv), BettingEngine.normalize_nba_team_name),
                    self.engine.build_game_edges(tennis_odds, tennis_stats, float(home_adv), self._normalize_tennis_player_name),
                )
            return edges_cache[key]

        def score(settings: dict) -> float:
            nonlocal eval_count
            eval_count += 1
            mlb_e, nba_e, ten_e = get_edges(settings["home_field_adv"])
            s = [self._score_settings_for_edges(e, settings, bankroll, only_later_today, selected_local_date)
                 for e in (mlb_e, nba_e, ten_e)]
            profit = sum(x["expected_profit"] for x in s)
            one_plus = 1 - float(np.prod([1 - x["at_least_one_win_prob"] for x in s]))
            balance_rate = max(x.get("balanced_game_rate", 0.0) for x in s)
            concentration = float(np.clip(1.0 - max(x.get("max_team_exposure_pct", 0.0) for x in s), 0.0, 1.0))
            if objective == "at_least_one_win_prob":
                return one_plus
            if objective == "balanced":
                return profit * (0.20 + 0.30 * one_plus + 0.20 * concentration + 0.30 * balance_rate)
            return profit

        best_score = score(best)
        for _ in range(2):
            if (perf_counter() - search_started) >= 12.0 or eval_count >= 90:
                break
            for param, candidates in search_space.items():
                if (perf_counter() - search_started) >= 12.0 or eval_count >= 90:
                    break
                local_best_val, local_best_score = best[param], best_score
                for val in candidates:
                    if (perf_counter() - search_started) >= 12.0 or eval_count >= 90:
                        break
                    cand = {**best, param: val}
                    s = score(cand)
                    if s > local_best_score:
                        local_best_score, local_best_val = s, val
                best[param] = local_best_val
                best_score = local_best_score
        return best

    # ── CSS & visual components ───────────────────────────────────────────────

    @staticmethod
    def _inject_styles() -> None:
        st.markdown("""
<style>
/* ── global ──────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}
.main .block-container { padding-top: 1.5rem; }

h1 {
    background: linear-gradient(90deg, #58a6ff 0%, #bc8cff 60%, #ff7b72 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important; font-weight: 800 !important;
    letter-spacing: -0.5px;
}
h2, h3 { color: #c9d1d9 !important; font-weight: 700 !important; }

/* ── pick cards ──────────────────────────────────────── */
.picks-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.pick-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 16px 18px 14px;
    flex: 1 1 160px; min-width: 160px; max-width: 220px;
    position: relative;
    transition: transform 0.15s, box-shadow 0.15s;
}
.pick-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
.pick-sport-badge {
    position: absolute; top: 10px; right: 12px;
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #8b949e;
}
.pick-game  { font-size: 0.72rem; color: #8b949e; margin-bottom: 6px; line-height: 1.3; }
.pick-team  { font-size: 1.05rem; font-weight: 700; color: #e6edf3; margin-bottom: 4px; }
.pick-odds  { font-size: 1.5rem; font-weight: 800; line-height: 1; }
.odds-pos   { color: #3fb950; }
.odds-neg   { color: #79c0ff; }
.pick-badges { margin-top: 8px; display: flex; gap: 4px; flex-wrap: wrap; }
.badge {
    display: inline-block; padding: 2px 9px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 700;
}
.badge-green  { background: #1a4731; color: #3fb950; border: 1px solid #238636; }
.badge-yellow { background: #3d2e00; color: #e3b341; border: 1px solid #9e6a03; }
.badge-red    { background: #2d1a1a; color: #f78166; border: 1px solid #8b1a1a; }
.badge-blue   { background: #1a2c3d; color: #58a6ff; border: 1px solid #1f6feb; }
.pick-book    { font-size: 0.68rem; color: #6e7681; margin-top: 5px; }

/* ── section label ────────────────────────────────────── */
.section-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #8b949e;
    border-bottom: 1px solid #21262d; padding-bottom: 6px;
    margin: 24px 0 12px;
}

/* ── stat bar (sidebar) ───────────────────────────────── */
.stat-row {
    display: flex; justify-content: space-between;
    font-size: 0.8rem; color: #8b949e; margin-bottom: 4px;
}
.stat-val { font-weight: 700; color: #c9d1d9; }

/* ── metric overrides ────────────────────────────────── */
[data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #8b949e !important; }
[data-testid="stMetricDelta"] { font-size: 0.72rem !important; }

/* ── buttons ─────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important; font-weight: 600 !important;
    transition: all 0.15s !important; letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.45) !important;
}

/* ── tabs ────────────────────────────────────────────── */
[data-testid="stTabs"] button { font-weight: 600 !important; font-size: 0.9rem !important; }

/* ── dataframe ───────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── expander ────────────────────────────────────────── */
[data-testid="stExpander"] details {
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    background: #161b22 !important;
}
</style>
""", unsafe_allow_html=True)

    @staticmethod
    def _pick_card_html(team: str, game: str, odds: float, edge_pct: float, ev: float, book: str, sport: str = "") -> str:
        odds_str = f"+{int(odds)}" if odds >= 0 else str(int(odds))
        odds_cls = "odds-pos" if odds >= 0 else "odds-neg"
        if edge_pct >= _EDGE_HIGH:
            e_cls, e_lbl = "badge-green", f"Edge {edge_pct:.1f}%"
        elif edge_pct >= _EDGE_MED:
            e_cls, e_lbl = "badge-yellow", f"Edge {edge_pct:.1f}%"
        else:
            e_cls, e_lbl = "badge-red", f"Edge {edge_pct:.1f}%"
        ev_lbl = f"EV {ev:+.3f}"
        sport_badge = f'<div class="pick-sport-badge">{sport}</div>' if sport else ""
        short_game = game[:30] + "…" if len(game) > 30 else game
        return f"""<div class="pick-card">
  {sport_badge}
  <div class="pick-game">{short_game}</div>
  <div class="pick-team">{team}</div>
  <div class="pick-odds {odds_cls}">{odds_str}</div>
  <div class="pick-badges">
    <span class="badge {e_cls}">{e_lbl}</span>
    <span class="badge badge-blue">{ev_lbl}</span>
  </div>
  <div class="pick-book">{book}</div>
</div>"""

    def _render_top_picks_row(self, edges_df: pd.DataFrame, sport: str = "", n: int = 6) -> bool:
        if edges_df.empty:
            return False
        top = edges_df.head(n)
        cards_html = "".join(
            self._pick_card_html(
                team=str(r["team"]),
                game=str(r["game"]),
                odds=float(r["american_odds"]),
                edge_pct=float(r["edge"]) * 100,
                ev=float(r["model_prob"]) * float(r["decimal_odds"]) - 1,
                book=str(r.get("book", "")),
                sport=sport,
            )
            for _, r in top.iterrows()
        )
        st.markdown(f'<div class="picks-grid">{cards_html}</div>', unsafe_allow_html=True)
        return True

    @staticmethod
    def _color_edge(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
        def _cell(val):
            try:
                v = float(val)
            except Exception:
                return ""
            if v >= _EDGE_HIGH:
                return "background-color:#1a4731;color:#3fb950;font-weight:700"
            if v >= _EDGE_MED:
                return "background-color:#3d2e00;color:#e3b341;font-weight:600"
            return ""
        try:
            return df.style.map(_cell, subset=["Edge %"])
        except AttributeError:
            return df.style.applymap(_cell, subset=["Edge %"])

    # ── league section ────────────────────────────────────────────────────────

    def _render_league_section(
        self,
        league_label: str,
        edges_df: pd.DataFrame,
        min_edge: float,
        only_later_today: bool,
        selected_local_date: date | None,
        bankroll: float,
        max_alloc_parlays: int,
        fractional_kelly: float,
        diversification_bias: float,
        max_team_uses: int,
        max_team_exposure_pct: float,
        contrarian_mix_pct: float,
        max_bet_pct: float,
        parlay_legs: int,
        parlay_top_n: int,
    ) -> None:
        if edges_df.empty:
            st.info(f"No {league_label.lower()} moneyline games found right now.")
            return

        # date filter
        if selected_local_date is not None:
            edges_df = BettingEngine.filter_by_local_date(edges_df, selected_local_date)
            if edges_df.empty:
                st.info(f"No {league_label.lower()} games on {selected_local_date.isoformat()}.")
                return

        local_today = datetime.now().astimezone().date()
        if only_later_today and (selected_local_date is None or selected_local_date == local_today):
            edges_df = BettingEngine.filter_later_today(edges_df)
            if edges_df.empty:
                st.info(f"No {league_label.lower()} games remaining today.")
                return

        # schedule
        schedule = edges_df[["game", "start_utc"]].drop_duplicates(subset=["game"]).sort_values("start_utc").copy()
        local_tz = datetime.now().astimezone().tzinfo
        schedule["Start"] = pd.to_datetime(schedule["start_utc"], utc=True).dt.tz_convert(local_tz).dt.strftime("%b %d  %I:%M %p %Z")
        with st.expander(f"Schedule — {len(schedule)} games", expanded=False):
            st.dataframe(schedule[["game", "Start"]].rename(columns={"game": "Game"}), use_container_width=True, hide_index=True)

        min_win_prob = float(st.session_state.get("min_win_prob", 50.0))
        filtered = BettingEngine.filter_realistic(edges_df, min_edge, min_win_prob)

        # ── top picks ──
        st.markdown('<div class="section-label">Top Picks</div>', unsafe_allow_html=True)
        if filtered.empty:
            st.info(f"No picks with edge ≥ {min_edge:.1f}% and win chance ≥ {min_win_prob:.0f}%. Lower either minimum in the sidebar.")
            return

        self._render_top_picks_row(filtered, sport=league_label, n=6)

        # ── pick table ──
        st.markdown(f'<div class="section-label">All Qualifying Picks &nbsp;·&nbsp; {len(filtered)} found</div>', unsafe_allow_html=True)

        base_cols = ["game", "team", "book", "books", "american_odds", "model_prob", "implied_prob_no_vig", "edge"]
        table = filtered[base_cols].copy()
        table["model_prob"] = (table["model_prob"] * 100).round(2)
        table["implied_prob_no_vig"] = (table["implied_prob_no_vig"] * 100).round(2)
        table["edge"] = (table["edge"] * 100).round(2)
        table.columns = ["Game", "Team", "Book", "Books", "Odds", "Model %", "Market % (No Vig)", "Edge %"]
        st.dataframe(self._color_edge(table), use_container_width=True, hide_index=True)

        # context columns (baseball-only)
        ctx_cols = [c for c in [
            "home_pitcher", "away_pitcher", "home_pitcher_era", "away_pitcher_era",
            "home_bullpen_ip_last3", "away_bullpen_ip_last3",
            "home_lineup_confirmed", "away_lineup_confirmed",
            "home_lineup_ops", "away_lineup_ops",
            "weather_condition", "weather_temp_f",
        ] if c in filtered.columns]
        if ctx_cols:
            with st.expander("Game context — pitchers, lineups, weather", expanded=False):
                ctx = filtered[["game", "team"] + ctx_cols].copy()
                for c in ["home_lineup_confirmed", "away_lineup_confirmed"]:
                    if c in ctx.columns:
                        ctx[c] = ctx[c].map({True: "Yes", False: "No"}).fillna("No")
                for c in ["home_bullpen_ip_last3", "away_bullpen_ip_last3", "home_lineup_ops", "away_lineup_ops"]:
                    if c in ctx.columns:
                        ctx[c] = pd.to_numeric(ctx[c], errors="coerce").round(2)
                for c in ["home_pitcher_era", "away_pitcher_era"]:
                    if c in ctx.columns:
                        ctx[c] = pd.to_numeric(ctx[c], errors="coerce").round(2)
                st.dataframe(ctx, use_container_width=True, hide_index=True)

        # ── parlay builder ──
        st.markdown('<div class="section-label">Parlay Builder</div>', unsafe_allow_html=True)

        seed_games = (
            filtered.sort_values("pick_score", ascending=False)
            .head(parlay_top_n)["game"].drop_duplicates().tolist()
        )
        parlay_input = edges_df[edges_df["game"].isin(seed_games)].copy()
        parlay_df = self.engine.build_parlays(parlay_input, max_legs=parlay_legs, top_n=parlay_top_n)

        if parlay_df.empty:
            st.info("Not enough qualifying picks to build parlays.")
            return

        st.caption("Built from both sides of your top games — allows hedge/failsafe combinations.")
        display_out = parlay_df[["legs", "parlay", "decimal_odds", "implied_prob", "model_prob", "edge", "ev_per_1u"]].copy()
        display_out["decimal_odds"] = display_out["decimal_odds"].round(2)
        display_out["implied_prob"] = (display_out["implied_prob"] * 100).round(2)
        display_out["model_prob"] = (display_out["model_prob"] * 100).round(2)
        display_out["edge"] = (display_out["edge"] * 100).round(2)
        display_out["ev_per_1u"] = display_out["ev_per_1u"].round(3)
        display_out.columns = ["Legs", "Parlay", "Dec Odds", "Implied %", "Model %", "Edge %", "EV / 1u"]
        st.dataframe(display_out.head(30), use_container_width=True, hide_index=True)

        # ── bankroll allocation ──
        st.markdown('<div class="section-label">Bankroll Allocation Plan</div>', unsafe_allow_html=True)

        alloc_df = self.engine.allocate_bankroll_to_parlays(
            parlay_df=parlay_df,
            bankroll=float(bankroll),
            max_parlays=int(max_alloc_parlays),
            max_bet_pct=float(max_bet_pct) / 100,
            fractional_kelly=float(fractional_kelly),
            diversification_bias=float(diversification_bias),
            max_team_uses=int(max_team_uses),
            max_team_exposure_pct=float(max_team_exposure_pct) / 100,
            contrarian_mix_pct=float(contrarian_mix_pct) / 100,
        )

        if alloc_df.empty:
            st.info("No positive-EV parlays available with current settings. Try lowering the edge threshold or switching to the Balanced preset.")
            return

        risk = self.engine.summarize_allocation_risk(alloc_df)
        allocated_pct = risk["total_stake"] / bankroll * 100 if bankroll > 0 else 0.0
        team_exp_pct = risk["max_team_exposure"] / bankroll * 100 if bankroll > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bankroll", f"${bankroll:,.2f}")
        c2.metric("Allocated", f"${risk['total_stake']:,.2f}", delta=f"{allocated_pct:.0f}% of roll")
        c3.metric("Expected Profit", f"${risk['expected_profit']:,.2f}")
        c4.metric("Win 1+ Parlays", f"{risk['at_least_one_win_prob'] * 100:.1f}%")

        c5, c6, c7 = st.columns(3)
        c5.metric("Max Payout", f"${risk['total_payout_if_all_win']:,.2f}")
        c6.metric("Max Profit", f"${risk['total_profit_if_all_win']:,.2f}")
        c7.metric("Lose All Prob", f"{risk['lose_all_prob'] * 100:.1f}%")

        if team_exp_pct > 40:
            st.warning(f"Highest single-team exposure: ${risk['max_team_exposure']:,.2f} ({team_exp_pct:.1f}% of bankroll) — consider lowering 'Max team exposure' in Advanced settings.")
        else:
            st.success(f"Single-team exposure is healthy: ${risk['max_team_exposure']:,.2f} ({team_exp_pct:.1f}% of bankroll)")

        st.caption("Win % uses model probability per parlay. Lose-all probability assumes independent parlay outcomes.")

        display_alloc = alloc_df[["legs", "parlay", "stake", "kelly_fraction", "win_loss_pct", "decimal_odds", "potential_payout", "expected_profit", "ev_per_1u"]].copy()
        display_alloc["kelly_fraction"] = (display_alloc["kelly_fraction"] * 100).round(1)
        display_alloc["win_loss_pct"] = display_alloc["win_loss_pct"].round(1)
        display_alloc["stake"] = display_alloc["stake"].round(2)
        display_alloc["decimal_odds"] = display_alloc["decimal_odds"].round(2)
        display_alloc["potential_payout"] = display_alloc["potential_payout"].round(2)
        display_alloc["expected_profit"] = display_alloc["expected_profit"].round(2)
        display_alloc["ev_per_1u"] = display_alloc["ev_per_1u"].round(3)
        display_alloc.columns = ["Legs", "Parlay", "Stake ($)", "Kelly %", "Win %", "Dec Odds", "Could Win ($)", "Exp Profit ($)", "EV / $1"]
        st.dataframe(display_alloc, use_container_width=True, hide_index=True)

        with st.expander("Team exposure breakdown", expanded=False):
            team_exp = (
                alloc_df.assign(tl=alloc_df["teams"]).explode("tl")
                .groupby("tl", as_index=False)
                .agg(parlays=("tl", "size"), stake=("stake", "sum"))
                .sort_values(["parlays", "stake"], ascending=False)
            )
            team_exp.columns = ["Team", "Parlays", "Total Stake ($)"]
            team_exp["Total Stake ($)"] = team_exp["Total Stake ($)"].round(2)
            st.dataframe(team_exp, use_container_width=True, hide_index=True)
            over = team_exp[team_exp["Parlays"] > 2]
            if not over.empty:
                st.warning(f"{over.iloc[0]['Team']} appears in {int(over.iloc[0]['Parlays'])} parlays — correlated loss risk.")

        # ── track for simulation ──
        with st.expander("Track parlays & save for later", expanded=False):
            sim_date = selected_local_date or datetime.now().astimezone().date()

            st.markdown("**Save parlay ideas**")
            sel = parlay_df.head(20)[["legs", "parlay", "games", "teams", "decimal_odds", "model_prob", "ev_per_1u"]].copy()
            sel.insert(0, "save", False)
            sel["decimal_odds"] = sel["decimal_odds"].round(2)
            sel["model_prob"] = (sel["model_prob"] * 100).round(2)
            sel["ev_per_1u"] = sel["ev_per_1u"].round(3)
            edited = st.data_editor(
                sel, use_container_width=True, hide_index=True,
                disabled=["legs", "parlay", "games", "teams", "decimal_odds", "model_prob", "ev_per_1u"],
                column_config={"save": st.column_config.CheckboxColumn("Save")},
                key=f"save_parlays_{league_label.lower()}",
            )
            if st.button(f"Save selected {league_label} parlays", key=f"save_btn_{league_label.lower()}"):
                chosen = edited[edited["save"]].copy()
                if chosen.empty:
                    st.info("Tick at least one parlay first.")
                else:
                    self._save_parlays([{
                        "saved_at": datetime.now().isoformat(timespec="seconds"),
                        "league": league_label, "legs": int(r["legs"]),
                        "parlay": r["parlay"], "games": list(r["games"]),
                        "teams": list(r["teams"]), "decimal_odds": float(r["decimal_odds"]),
                        "model_prob": float(r["model_prob"]) / 100.0, "ev_per_1u": float(r["ev_per_1u"]),
                    } for _, r in chosen.iterrows()])
                    st.success(f"Saved {len(chosen)} parlay(s).")

            saved = self._load_saved_parlays()
            if saved:
                saved_df = pd.DataFrame(saved)
                cols = [c for c in ["saved_at", "league", "legs", "parlay", "decimal_odds", "model_prob", "ev_per_1u"] if c in saved_df.columns]
                if cols:
                    sv = saved_df[cols].copy()
                    if "model_prob" in sv.columns:
                        sv["model_prob"] = (sv["model_prob"] * 100).round(2)
                    if "decimal_odds" in sv.columns:
                        sv["decimal_odds"] = sv["decimal_odds"].round(2)
                    if "ev_per_1u" in sv.columns:
                        sv["ev_per_1u"] = sv["ev_per_1u"].round(3)
                    st.markdown("**Previously saved**")
                    st.dataframe(sv, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("**Add recommended allocation to Simulation**")
            reco = alloc_df.head(20)[["legs", "parlay", "games", "teams", "event_ids", "sport_keys", "decimal_odds", "stake", "ev_per_1u"]].copy()
            reco.insert(0, "track", True)
            for c in ["stake", "decimal_odds", "ev_per_1u"]:
                reco[c] = pd.to_numeric(reco[c], errors="coerce").round(3 if c == "ev_per_1u" else 2)
            reco_edited = st.data_editor(
                reco, use_container_width=True, hide_index=True,
                disabled=["legs", "parlay", "games", "teams", "event_ids", "sport_keys", "decimal_odds", "ev_per_1u"],
                column_config={
                    "track": st.column_config.CheckboxColumn("Track"),
                    "stake": st.column_config.NumberColumn("Stake (units)", min_value=0.1, step=0.1, format="%.2f"),
                    "ev_per_1u": st.column_config.NumberColumn("EV / 1u", format="%.3f"),
                },
                key=f"track_reco_{league_label.lower()}",
            )
            if st.button(f"Add {league_label} parlays to Simulation", key=f"track_reco_btn_{league_label.lower()}"):
                chosen = reco_edited[reco_edited["track"]].copy()
                if chosen.empty:
                    st.info("Select at least one parlay.")
                else:
                    tickets = []
                    for _, row in chosen.iterrows():
                        eids = [str(e) for e in list(row["event_ids"]) if e is not None and str(e).strip()]
                        skeys = [str(s) for s in list(row["sport_keys"]) if s is not None and str(s).strip()]
                        if not eids:
                            continue
                        tickets.append({
                            "created_at": datetime.now().isoformat(timespec="microseconds"),
                            "game_date_local": sim_date.isoformat(),
                            "league": league_label, "legs": int(row["legs"]),
                            "parlay": row["parlay"], "games": list(row["games"]),
                            "teams": list(row["teams"]), "event_ids": eids,
                            "sport_keys": skeys, "decimal_odds": float(row["decimal_odds"]),
                            "stake": float(row["stake"]), "status": "pending",
                        })
                    added = self._append_simulation_tickets(tickets)
                    st.success(f"Added {added} parlay(s) to Simulation.") if added else st.info("No valid parlays added.")

    # ── simulation tab ────────────────────────────────────────────────────────

    def _render_simulation_tab(self, client: OddsAPIClient, selected_local_date: date) -> None:
        st.subheader("Parlay Simulation Tracker")

        tickets = self._load_simulation_tickets()
        if not tickets:
            st.info("No tracked parlays yet. Use 'Track parlays & save for later' in each sport tab.")
            return

        col_grade, col_clear, col_toggle = st.columns([1, 1, 2])
        with col_grade:
            if st.button("Grade parlays now", use_container_width=True):
                settled, _ = self._grade_simulation_tickets(client)
                msg = f"Graded {settled} parlay(s)." if settled else "No newly completed parlays found yet."
                st.success(msg) if settled else st.info(msg)
                tickets = self._load_simulation_tickets()
        with col_clear:
            if st.button("Clear all simulations", use_container_width=True, type="secondary"):
                st.session_state["_confirm_clear_sim"] = True
        if st.session_state.get("_confirm_clear_sim"):
            st.warning("This will permanently delete all tracked parlays. Are you sure?")
            yes_col, no_col = st.columns(2)
            if yes_col.button("Yes, clear everything", type="primary", use_container_width=True):
                self._save_simulation_tickets([])
                st.session_state.pop("_confirm_clear_sim", None)
                st.success("All simulations cleared.")
                st.rerun()
            if no_col.button("Cancel", use_container_width=True):
                st.session_state.pop("_confirm_clear_sim", None)
                st.rerun()
        with col_toggle:
            show_all = st.checkbox("Show all dates (cumulative)", value=True)

        df = pd.DataFrame(tickets)
        if df.empty:
            st.info("No tracked parlays.")
            return

        if "status" not in df.columns:
            df["status"] = "pending"

        # all-time stats
        all_settled = df[df["status"] == "settled"].copy()
        all_stake = float(pd.to_numeric(df.get("stake", 0), errors="coerce").fillna(0).sum())
        all_profit = float(pd.to_numeric(all_settled.get("profit_u", 0), errors="coerce").fillna(0).sum()) if not all_settled.empty else 0.0
        all_roi = (all_profit / all_stake * 100) if all_stake > 0 else 0.0
        all_wins = int((all_settled.get("result") == "won").sum()) if "result" in all_settled.columns else 0
        all_losses = int((all_settled.get("result") == "lost").sum()) if "result" in all_settled.columns else 0

        st.markdown('<div class="section-label">All-Time Record</div>', unsafe_allow_html=True)
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Total Tracked", f"{len(df)}")
        a2.metric("Settled", f"{len(all_settled)}")
        a3.metric("Record", f"{all_wins}W – {all_losses}L")
        a4.metric("Profit (units)", f"{all_profit:+.2f}", delta="above breakeven" if all_profit > 0 else "below breakeven")
        a5.metric("ROI", f"{all_roi:+.1f}%")

        if not show_all and "game_date_local" in df.columns:
            df = df[df["game_date_local"] == selected_local_date.isoformat()].copy()
            if df.empty:
                st.info(f"No tracked parlays for {selected_local_date.isoformat()}.")
                return

            settled_df = df[df["status"] == "settled"].copy()
            day_stake = float(pd.to_numeric(df.get("stake", 0), errors="coerce").fillna(0).sum())
            day_profit = float(pd.to_numeric(settled_df.get("profit_u", 0), errors="coerce").fillna(0).sum()) if not settled_df.empty else 0.0
            day_roi = (day_profit / day_stake * 100) if day_stake > 0 else 0.0
            d_wins = int((settled_df.get("result") == "won").sum()) if "result" in settled_df.columns else 0
            d_losses = int((settled_df.get("result") == "lost").sum()) if "result" in settled_df.columns else 0

            st.markdown(f'<div class="section-label">{selected_local_date.isoformat()}</div>', unsafe_allow_html=True)
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Tracked", f"{len(df)}")
            d2.metric("Record", f"{d_wins}W – {d_losses}L")
            d3.metric("Profit (units)", f"{day_profit:+.2f}")
            d4.metric("ROI", f"{day_roi:+.1f}%")

        show_cols = [c for c in [
            "created_at", "game_date_local", "league", "legs", "parlay",
            "stake", "decimal_odds", "status", "result",
            "failed_legs", "failed_legs_detail", "profit_u", "settled_at",
        ] if c in df.columns]
        out = df[show_cols].copy()
        for c, fn in [("stake", 2), ("decimal_odds", 2), ("profit_u", 3)]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(fn)
        if "failed_legs" in out.columns:
            out["failed_legs"] = pd.to_numeric(out["failed_legs"], errors="coerce").fillna(0).astype(int)
        st.dataframe(out, use_container_width=True, hide_index=True)

        st.divider()
        self._render_backtest_section(st.session_state.get("api_key", ""))

    # ── backtest section ──────────────────────────────────────────────────────

    def _render_backtest_section(self, api_key: str) -> None:
        with st.expander("📈 Historical Backtest Generator", expanded=False):
            st.caption(
                "Replay past N days using historical odds snapshots + final MLB scores. "
                "Each day costs 1 Odds API quota request."
            )

            col_days, col_edge, col_parlay = st.columns([2, 2, 1])
            with col_days:
                n_days = st.slider(
                    "Days to simulate", min_value=1, max_value=30, value=7,
                    key="bt_n_days",
                    help="Simulates the N days ending yesterday.",
                )
            with col_edge:
                bt_edge = st.slider(
                    "Min edge (%)", min_value=0.0, max_value=15.0, value=2.0, step=0.5,
                    key="bt_min_edge",
                )
            with col_parlay:
                bt_parlays = st.checkbox("Parlays", value=False, key="bt_parlays",
                                         help="Also simulate parlay allocation each day.")

            bt_bankroll = 10.0
            bt_max_parlays = 5
            if bt_parlays:
                pc1, pc2 = st.columns(2)
                with pc1:
                    bt_bankroll = st.number_input(
                        "Parlay bankroll ($)", min_value=1.0, max_value=10000.0,
                        value=10.0, step=1.0, key="bt_bankroll",
                    )
                with pc2:
                    bt_max_parlays = st.number_input(
                        "Max parlays/day", min_value=1, max_value=15,
                        value=5, key="bt_max_parlays",
                    )

            if not api_key:
                st.warning("Enter your Odds API key in the sidebar to run backtests.")
                return

            run_btn = st.button(
                f"▶  Run {n_days}-day backtest", key="bt_run_btn", type="primary",
                use_container_width=True,
            )

            if run_btn:
                # Always wipe previous results before a new run so stale data never leaks through.
                st.session_state.pop("_bt_results", None)
                st.session_state.pop("_bt_auth_error", None)

                yesterday = (datetime.now().astimezone().date() - timedelta(days=1))
                progress_bar = st.progress(0.0)
                status_txt = st.empty()

                def _cb(step: int, total: int, d) -> None:
                    progress_bar.progress(step / total)
                    status_txt.text(f"Simulating {d.isoformat()}  ({step}/{total})")

                try:
                    day_results, summary = run_backtest(
                        end_date=yesterday,
                        n_days=n_days,
                        api_key=api_key,
                        min_edge_pct=float(bt_edge),
                        build_parlays=bool(bt_parlays),
                        bankroll=float(bt_bankroll),
                        max_parlays=int(bt_max_parlays),
                        progress_callback=_cb,
                    )
                    progress_bar.progress(1.0)
                    status_txt.empty()
                    st.session_state["_bt_results"] = (day_results, summary)
                except HistoricalOddsUnavailable as e:
                    progress_bar.empty()
                    status_txt.empty()
                    st.session_state["_bt_auth_error"] = str(e)
                except Exception as e:
                    progress_bar.empty()
                    status_txt.empty()
                    st.error(f"Backtest error: {e}")
                    return

            if "_bt_auth_error" in st.session_state:
                st.error(
                    "**Historical odds unavailable with your current API plan.**\n\n"
                    + st.session_state["_bt_auth_error"]
                    + "\n\nUpgrade at [the-odds-api.com](https://the-odds-api.com) "
                    "to unlock historical snapshots."
                )
                return

            if "_bt_results" not in st.session_state:
                return

            day_results, summary = st.session_state["_bt_results"]

            # Detect old cached results where every day was skipped due to a 401.
            # (Can happen when session state persists from a run before the fix.)
            if all(r.get("skipped") for r in day_results):
                first_reason = day_results[0].get("reason", "") if day_results else ""
                if any(tok in first_reason for tok in ("401", "Unauthorized", "paid", "subscription")):
                    st.error(
                        "**Historical odds unavailable with your current API plan.**\n\n"
                        "The Odds API historical endpoint requires a paid subscription. "
                        "Free-tier keys only support live odds.\n\n"
                        "Upgrade at [the-odds-api.com](https://the-odds-api.com) "
                        "to unlock historical snapshots."
                    )
                else:
                    st.warning(f"All days were skipped. Reason: {first_reason or 'unknown'}")
                return

            # ── summary metrics ───────────────────────────────────────────────
            st.markdown(
                f'<div class="section-label">Backtest: {summary["start_date"]} → {summary["end_date"]}'
                f' &nbsp;·&nbsp; {summary["active_days"]} active days'
                f' &nbsp;·&nbsp; min edge {summary["min_edge_pct"]}%</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total picks", summary["total_picks"])
            m2.metric(
                "Record",
                f"{summary['total_wins']}W – {summary['total_losses']}L",
            )
            m3.metric("Win rate", f"{summary['win_rate_pct']:.1f}%")
            m4.metric(
                "Total profit",
                f"{summary['total_profit_u']:+.2f}u",
                delta="profitable" if summary["total_profit_u"] > 0 else "unprofitable",
            )
            m5.metric("ROI", f"{summary['total_roi_pct']:+.1f}%")

            if "total_parlay_count" in summary:
                st.markdown('<div class="section-label">Parlay Simulation</div>', unsafe_allow_html=True)
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Parlays", f"{summary['total_parlay_wins']}/{summary['total_parlay_count']}")
                p2.metric("Staked", f"${summary['total_parlay_staked']:.2f}")
                p3.metric("Profit", f"{summary['total_parlay_profit']:+.2f}u")
                p4.metric("Parlay ROI", f"{summary['total_parlay_roi_pct']:+.1f}%")

            # ── cumulative profit chart ───────────────────────────────────────
            running = summary.get("running_profit", [])
            if running:
                chart_df = pd.DataFrame(running).set_index("date")
                st.line_chart(chart_df, y="cumulative_profit_u", use_container_width=True, height=160)

            # ── per-day table ─────────────────────────────────────────────────
            table_rows = []
            for r in day_results:
                if r.get("skipped"):
                    table_rows.append({
                        "date": r["date"],
                        "picks": "—",
                        "record": f"skipped ({r.get('reason', '')})",
                        "profit_u": "—",
                        "roi_pct": "—",
                        "avg_edge_pct": "—",
                    })
                else:
                    row = {
                        "date": r["date"],
                        "picks": r["picks"],
                        "record": f"{r['wins']}W – {r['losses']}L",
                        "profit_u": f"{r['profit_u']:+.3f}",
                        "roi_pct": f"{r['roi_pct']:+.1f}%",
                        "avg_edge_pct": f"{r['avg_edge_pct']:.2f}%",
                    }
                    if "parlay_count" in r:
                        row["parlays"] = f"{r['parlay_wins']}/{r['parlay_count']}"
                        row["parlay_profit"] = f"{r['parlay_profit']:+.2f}"
                    table_rows.append(row)

            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            # ── per-day pick detail ───────────────────────────────────────────
            active_days_list = [r for r in day_results if not r.get("skipped")]
            if active_days_list:
                with st.expander("Pick-by-pick detail", expanded=False):
                    all_picks = []
                    for r in active_days_list:
                        picks_df = r.get("picks_df")
                        if picks_df is not None and not picks_df.empty:
                            pf = picks_df.copy()
                            pf.insert(0, "date", r["date"])
                            pf["edge"] = (pf["edge"] * 100).round(2)
                            pf["model_prob"] = (pf["model_prob"] * 100).round(1)
                            pf["profit_u"] = pf["profit_u"].round(3)
                            all_picks.append(pf)
                    if all_picks:
                        combined = pd.concat(all_picks, ignore_index=True)
                        st.dataframe(combined, use_container_width=True, hide_index=True)

    # ── main ─────────────────────────────────────────────────────────────────

    def run(self) -> None:
        st.set_page_config(
            page_title="BetEdge — MLB · NBA · Tennis",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self._inject_styles()

        st.title("BetEdge by Herchelle")
        st.caption("Live edge finder for MLB, NBA & Tennis · educational model only, not financial advice")

        try:
            secrets_key = st.secrets.get("ODDS_API_KEY", "")
        except Exception:
            secrets_key = ""

        defaults = {
            "api_key": secrets_key, "season": datetime.now().year,
            "min_edge": 2.0, "min_win_prob": 50.0, "home_field_adv": 0.06, "parlay_legs": 3, "parlay_top_n": 8,
            "bankroll": 10.0, "max_alloc_parlays": 12, "fractional_kelly": 0.35,
            "diversification_bias": 0.85, "max_team_uses": 2, "max_bet_pct": 15.0,
            "max_team_exposure_pct": 35.0, "contrarian_mix_pct": 20.0,
            "game_day_offset": 0, "only_later_today": True,
        }

        pending_opt = st.session_state.pop("_pending_optimized_settings", None)
        pending_msg = st.session_state.pop("_pending_optimized_message", None)
        if isinstance(pending_opt, dict):
            for k, v in pending_opt.items():
                st.session_state[k] = v
            st.session_state["_show_opt_msg"] = pending_msg or "Settings applied."
            st.session_state["_optimizer_ran"] = pending_msg or "Settings applied."
            st.session_state.pop("max_alloc_parlays_quick", None)

        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        # ── SIDEBAR ──────────────────────────────────────────────────────────
        with st.sidebar:
            st.markdown("## Settings")

            # presets
            presets = {
                "Aggressive": {
                    "parlay_legs": 4, "parlay_top_n": 12, "max_alloc_parlays": 14,
                    "fractional_kelly": 0.70, "diversification_bias": 0.35, "max_team_uses": 4,
                    "max_team_exposure_pct": 60.0, "max_bet_pct": 30.0, "min_edge": 1.0,
                },
                "Balanced": {
                    "parlay_legs": 3, "parlay_top_n": 8, "max_alloc_parlays": 10,
                    "fractional_kelly": 0.40, "diversification_bias": 0.90, "max_team_uses": 2,
                    "max_team_exposure_pct": 35.0, "max_bet_pct": 15.0, "min_edge": 2.0,
                },
                "Conservative": {
                    "parlay_legs": 2, "parlay_top_n": 6, "max_alloc_parlays": 6,
                    "fractional_kelly": 0.20, "diversification_bias": 1.60, "max_team_uses": 1,
                    "max_team_exposure_pct": 20.0, "max_bet_pct": 8.0, "min_edge": 3.0,
                },
            }
            st.caption("Risk preset")
            p1, p2, p3 = st.columns(3)
            if p1.button("Aggressive", use_container_width=True):
                st.session_state["_pending_optimized_settings"] = presets["Aggressive"]
                st.session_state["_pending_optimized_message"] = "Aggressive preset applied."
                st.rerun()
            if p2.button("Balanced", use_container_width=True):
                st.session_state["_pending_optimized_settings"] = presets["Balanced"]
                st.session_state["_pending_optimized_message"] = "Balanced preset applied."
                st.rerun()
            if p3.button("Conservative", use_container_width=True):
                st.session_state["_pending_optimized_settings"] = presets["Conservative"]
                st.session_state["_pending_optimized_message"] = "Conservative preset applied."
                st.rerun()

            st.divider()

            st.text_input("Odds API Key", type="password", help="Get a free key at the-odds-api.com", key="api_key")

            today_local = datetime.now().astimezone().date()
            st.radio(
                "Game date",
                options=[0, 1],
                format_func=lambda x: f"Today ({today_local.isoformat()})" if x == 0 else f"Tomorrow ({(today_local + timedelta(days=1)).isoformat()})",
                horizontal=True, key="game_day_offset",
            )

            st.slider("Money to bet ($)", min_value=1.0, max_value=1000.0, step=1.0,
                      help="Total bankroll to allocate across recommended parlays.", key="bankroll")
            st.slider("Minimum edge (%)", min_value=0.0, max_value=20.0, step=0.5,
                      help="Only show picks where model probability exceeds market by this %.", key="min_edge")
            st.slider("Minimum win chance (%)", min_value=30.0, max_value=80.0, step=1.0,
                      help="Only show picks the model thinks win at least this often. 50 = model's predicted winner only.", key="min_win_prob")
            st.checkbox("Only games later today", key="only_later_today")

            st.divider()

            # optimizer
            st.caption("Auto-optimizer")
            o1, o2, o3 = st.columns(3)
            maximize_profit   = o1.button("Max $",    use_container_width=True, help="Maximize expected profit")
            maximize_wins     = o2.button("Max wins", use_container_width=True, help="Maximize 1+ parlay win probability")
            maximize_balanced = o3.button("Balance",  use_container_width=True, help="Balance profit and win probability")

            st.divider()

            with st.expander("Advanced settings"):
                st.number_input("Season", min_value=2020, max_value=2100, key="season")
                st.slider("Home field adjustment", min_value=-0.25, max_value=0.25, step=0.01,
                          help="Positive = boost home teams in the model.", key="home_field_adv")
                st.selectbox("Max parlay legs", options=[2, 3, 4], key="parlay_legs")
                st.slider("Top N picks for parlays", min_value=4, max_value=14, key="parlay_top_n")
                st.slider("Max parlays to allocate", min_value=2, max_value=20, key="max_alloc_parlays")
                st.slider("Fractional Kelly", min_value=0.10, max_value=1.0, step=0.05,
                          help="Lower = spread money more evenly, higher = concentrate on best picks.", key="fractional_kelly")
                st.slider("Diversification bias", min_value=0.0, max_value=2.0, step=0.05,
                          help="Higher = penalize parlays that reuse the same teams.", key="diversification_bias")
                st.slider("Max uses per team", min_value=1, max_value=6,
                          help="Limit how many parlays can include the same team.", key="max_team_uses")
                st.slider("Max team exposure (% of bankroll)", min_value=10.0, max_value=80.0, step=1.0,
                          help="Cap total dollars at risk on any single team across all parlays.", key="max_team_exposure_pct")
                st.slider("Opposing-side mix (%)", min_value=0.0, max_value=50.0, step=5.0,
                          help="Reserve some parlay slots for underdog or opposite-side combinations.", key="contrarian_mix_pct")
                st.slider("Max stake per parlay (% of bankroll)", min_value=5.0, max_value=50.0, step=1.0, key="max_bet_pct")

            if st.session_state.get("_show_opt_msg"):
                st.success(str(st.session_state.pop("_show_opt_msg")))

        # ── read settings ─────────────────────────────────────────────────────
        api_key               = str(st.session_state.get("api_key", ""))
        season                = int(st.session_state.get("season", datetime.now().year))
        min_edge              = float(st.session_state.get("min_edge", 2.0))
        home_field_adv        = float(st.session_state.get("home_field_adv", 0.06))
        parlay_legs           = int(st.session_state.get("parlay_legs", 3))
        parlay_top_n          = int(st.session_state.get("parlay_top_n", 8))
        bankroll              = float(st.session_state.get("bankroll", 10.0))
        max_alloc_parlays     = int(st.session_state.get("max_alloc_parlays", 10))
        fractional_kelly      = float(st.session_state.get("fractional_kelly", 0.35))
        diversification_bias  = float(st.session_state.get("diversification_bias", 0.85))
        max_team_uses         = int(st.session_state.get("max_team_uses", 1))
        max_bet_pct           = float(st.session_state.get("max_bet_pct", 15.0))
        max_team_exposure_pct = float(st.session_state.get("max_team_exposure_pct", 35.0))
        contrarian_mix_pct    = float(st.session_state.get("contrarian_mix_pct", 20.0))
        only_later_today      = bool(st.session_state.get("only_later_today", True))
        game_day_offset       = int(st.session_state.get("game_day_offset", 0))

        if not api_key:
            st.info("Enter your Odds API key in the sidebar to get started. Get a free key at the-odds-api.com")
            st.stop()

        # ── load data ─────────────────────────────────────────────────────────
        client = OddsAPIClient(api_key)
        try:
            with st.spinner("Fetching live odds and team stats…"):
                snapshot         = self.controller.load_sports(client, season, home_field_adv)
                baseball_odds    = snapshot.baseball["odds"]
                baseball_stats   = snapshot.baseball["stats"]
                baseball_context = snapshot.baseball["context"]
                baseball_edges   = snapshot.baseball["edges"]
                basketball_odds  = snapshot.basketball["odds"]
                basketball_stats = snapshot.basketball["stats"]
                basketball_edges = snapshot.basketball["edges"]
                tennis_odds      = snapshot.tennis["odds"]
                tennis_stats     = snapshot.tennis["stats"]
                tennis_edges     = snapshot.tennis["edges"]
        except requests.HTTPError as err:
            st.error(f"API error: {err}")
            st.stop()
        except Exception as err:
            st.error(f"Error loading data: {err}")
            st.stop()

        selected_local_date = datetime.now().astimezone().date() + timedelta(days=game_day_offset)

        # ── optimizer trigger ─────────────────────────────────────────────────
        if maximize_profit or maximize_wins or maximize_balanced:
            objective = "expected_profit" if maximize_profit else ("at_least_one_win_prob" if maximize_wins else "balanced")
            labels = {
                "expected_profit": "Optimized for maximum expected profit.",
                "at_least_one_win_prob": "Optimized for highest 1+ parlay win probability.",
                "balanced": "Optimized for balanced profit and win probability.",
            }
            with st.spinner("Running optimizer (up to 12 seconds)…"):
                best = self._optimize_settings(
                    baseball_odds=baseball_odds, baseball_stats=baseball_stats, baseball_context=baseball_context,
                    basketball_odds=basketball_odds, basketball_stats=basketball_stats,
                    tennis_odds=tennis_odds, tennis_stats=tennis_stats,
                    bankroll=bankroll, only_later_today=only_later_today,
                    selected_local_date=selected_local_date, objective=objective,
                )
            st.session_state["_pending_optimized_settings"] = best
            st.session_state["_pending_optimized_message"] = labels[objective]
            st.rerun()

        # ── post-optimizer parlay count slider ────────────────────────────────
        if st.session_state.get("_optimizer_ran"):
            st.markdown(
                '<p style="margin:0 0 4px 0; font-size:0.78rem; color:#8b949e; text-transform:uppercase; letter-spacing:.06em;">Parlay count — optimizer active</p>',
                unsafe_allow_html=True,
            )
            quick_count = st.slider(
                "Number of parlays to spread your bankroll across",
                min_value=2,
                max_value=20,
                value=int(st.session_state.get("max_alloc_parlays", 12)),
                step=1,
                key="max_alloc_parlays_quick",
                help="Drag to change how many parlays are built. Uses the current optimized settings — no need to re-run.",
            )
            max_alloc_parlays = quick_count

        # ── tabs ──────────────────────────────────────────────────────────────
        common = dict(
            min_edge=min_edge, only_later_today=only_later_today, selected_local_date=selected_local_date,
            bankroll=bankroll, max_alloc_parlays=max_alloc_parlays, fractional_kelly=fractional_kelly,
            diversification_bias=diversification_bias, max_team_uses=max_team_uses,
            max_team_exposure_pct=max_team_exposure_pct, contrarian_mix_pct=contrarian_mix_pct,
            max_bet_pct=max_bet_pct, parlay_legs=parlay_legs, parlay_top_n=parlay_top_n,
        )

        tab_bb, tab_nba, tab_ten, tab_sim = st.tabs(["⚾  Baseball", "🏀  Basketball", "🎾  Tennis", "📊  Simulation"])

        with tab_bb:
            self._render_league_section(league_label="Baseball", edges_df=baseball_edges, **common)

        with tab_nba:
            self._render_league_section(league_label="Basketball", edges_df=basketball_edges, **common)

        with tab_ten:
            self._render_league_section(league_label="Tennis", edges_df=tennis_edges, **common)

        with tab_sim:
            self._render_simulation_tab(client, selected_local_date)
