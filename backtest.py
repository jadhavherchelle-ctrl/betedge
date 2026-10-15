"""Multi-day MLB backtest generator (CLI).

Usage:
    python backtest.py                          # last 7 days, 2% edge
    python backtest.py --days 14
    python backtest.py --days 30 --min-edge 3.0
    python backtest.py --days 7 --parlay --bankroll 50
    python backtest.py --days 14 --csv results.csv
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from gambling_app.backtest_engine import run_backtest


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Period:      {summary['start_date']}  →  {summary['end_date']}")
    print(f"Days:        {summary['active_days']} active / {summary['n_days_requested']} total "
          f"({summary['skipped_days']} skipped)")
    print(f"Min edge:    {summary['min_edge_pct']}%")
    print()
    print("Straight picks")
    print(f"  Total:     {summary['total_picks']}")
    print(f"  Record:    {summary['total_wins']}W – {summary['total_losses']}L "
          f"({summary['win_rate_pct']:.1f}%)")
    print(f"  Profit:    {summary['total_profit_u']:+.3f}u")
    print(f"  ROI:       {summary['total_roi_pct']:+.2f}%")
    print(f"  Avg/day:   {summary['avg_profit_per_active_day']:+.3f}u")
    if "total_parlay_count" in summary:
        print()
        print("Parlays")
        print(f"  Count:     {summary['total_parlay_count']}  "
              f"({summary['total_parlay_wins']} won)")
        print(f"  Staked:    ${summary['total_parlay_staked']:.2f}")
        print(f"  Profit:    {summary['total_parlay_profit']:+.3f}u")
        print(f"  ROI:       {summary['total_parlay_roi_pct']:+.2f}%")
    print("=" * 60)


def save_csv(day_results: list[dict], path: str) -> None:
    rows = [{k: v for k, v in r.items() if k not in ("picks_df", "parlay_df")} for r in day_results]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSaved per-day results → {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-day MLB backtest generator.")
    p.add_argument("--days", type=int, default=7,
                   help="Number of past days to simulate (default: 7)")
    p.add_argument("--min-edge", type=float, default=2.0,
                   help="Minimum edge %% for qualifying picks (default: 2.0)")
    p.add_argument("--home-field-adv", type=float, default=0.06,
                   help="Home-field advantage factor (default: 0.06)")
    p.add_argument("--season", type=int, default=None,
                   help="MLB season year override (default: current year)")
    p.add_argument("--parlay", action="store_true",
                   help="Include parlay simulation")
    p.add_argument("--parlay-legs", type=int, default=3,
                   help="Max parlay legs (default: 3)")
    p.add_argument("--parlay-top-n", type=int, default=8,
                   help="Top N games considered for parlay combos (default: 8)")
    p.add_argument("--bankroll", type=float, default=10.0,
                   help="Bankroll for parlay allocation in $ (default: 10)")
    p.add_argument("--max-parlays", type=int, default=5,
                   help="Max parlays to allocate per day (default: 5)")
    p.add_argument("--csv", type=str, default=None,
                   help="Save per-day results to this CSV file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set the ODDS_API_KEY environment variable first.")
        raise SystemExit(2)

    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)

    def _progress(step: int, total: int, d) -> None:
        print(f"  [{step:2d}/{total}] {d.isoformat()}...", end=" ", flush=True)

    print(f"\nRunning {args.days}-day MLB backtest  (min edge {args.min_edge}%)\n")

    day_results, summary = run_backtest(
        end_date=yesterday,
        n_days=args.days,
        api_key=api_key,
        min_edge_pct=args.min_edge,
        home_field_adv=args.home_field_adv,
        season=args.season,
        build_parlays=args.parlay,
        parlay_legs=args.parlay_legs,
        parlay_top_n=args.parlay_top_n,
        bankroll=args.bankroll,
        max_parlays=args.max_parlays,
        progress_callback=_progress,
    )

    # Per-day inline output
    for r in day_results:
        if r.get("skipped"):
            print(f"  {r['date']}  SKIPPED — {r.get('reason', '')}")
        else:
            parlay_note = ""
            if "parlay_count" in r:
                parlay_note = (
                    f"  |  parlays {r['parlay_wins']}/{r['parlay_count']}"
                    f" {r['parlay_profit']:+.2f}u"
                )
            print(
                f"  {r['date']}  "
                f"{r['wins']}W/{r['losses']}L  "
                f"{r['profit_u']:+.3f}u  ROI {r['roi_pct']:+.1f}%"
                f"{parlay_note}"
            )

    print_summary(summary)

    if args.csv:
        save_csv(day_results, args.csv)

    raise SystemExit(0)
