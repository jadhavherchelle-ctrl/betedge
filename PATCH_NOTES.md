# Patch Notes

## 2026-04-26

### Added
- **Full bankroll deployment (water-filling allocation):** Bankroll allocation now uses an iterative water-fill loop instead of a single-pass normalization. Excess weight from parlays that hit the per-parlay stake cap is redistributed to the remaining parlays. Every dollar of the configured bankroll is always deployed — e.g., a $10 bankroll now always totals $10 across selected parlays.
- **Heavy-favourite protection:** After parlay selection, the engine scans the final portfolio for any team with American odds more negative than -140 (heavy favourite) that appears in 2+ parlays all on the same side. For each one found, up to 2 extra "hedge" parlays are appended that include the opposing team. These hedge slots sit outside the normal `max_parlays` limit so the main allocation is not reduced to make room.
- **Per-leg odds stored in parlay rows:** Each parlay now carries a `leg_odds` tuple with the American odds of every leg. This enables the heavy-favourite detection to read individual odds without re-joining to the edges dataframe.

### Changed
- **Hedge loop rounds increased:** The same-side imbalance replacement loop now runs up to 10 iterations (was 4), allowing more opposite-side swaps per allocation pass.
- **Hedge baseline bonus raised:** The scoring bonus applied to opposite-side hedge candidates increased from +0.22 to +0.40, making opposite-side parlays more competitive against same-side duplicates even when their EV is slightly lower.
- **UI full redesign:**
  - Dark GitHub-style theme with gradient title, card surfaces, and consistent color palette throughout.
  - Top-pick cards displayed at the top of each sport tab showing the highest-edge bets with color-coded edge badges (green ≥5%, yellow ≥2%), odds, EV, and book.
  - Edge % column in the picks table is color-coded (green / yellow) for instant visual scanning.
  - Sidebar reorganized: essential controls (bankroll, edge threshold, game date) always visible; advanced settings collapsed in an expander.
  - Game context (pitchers, lineups, weather), team exposure breakdown, and parlay tracking moved into collapsible expanders so allocation metrics stay front and center.
  - Auto-optimizer reduced to 3 compact buttons in the sidebar: **Max $**, **Max wins**, **Balance**.
  - Tab labels updated with sport emojis: ⚾ Baseball, 🏀 Basketball, 🎾 Tennis, 📊 Simulation.
  - Simulation tab shows all-time record, profit (units), and ROI as prominent metrics.
  - App title updated to **BetEdge by Herchelle**.

## 2026-04-19

### Added
- MLB model context upgrades integrated into scoring:
  - Bullpen context (`bullpen_ip_last3`, bullpen ERA/WHIP influence)
  - Lineup context (`lineup_confirmed`, lineup OPS influence)
  - Existing probable starter context remains active
- New replay utility script: `simulate_yesterday.py`
  - Pulls historical MLB odds snapshot for yesterday
  - Recomputes model edges with your current parameters
  - Grades picks versus final MLB scores and reports units/ROI
- Legal protection artifacts for authorship/ownership records:
  - `LEGAL_PROTECTION_PACKET.md`
  - `EVIDENCE_SNAPSHOT_2026-04-19.md`
  - `DISCLOSURE_LOG.md`
  - `legal/templates/NDA_TEMPLATE.md`
  - `legal/templates/DEMO_ACCESS_TERMS_TEMPLATE.md`
  - `legal/templates/OWNERSHIP_NO_USE_MESSAGE_TEMPLATE.md`
  - `legal/templates/LEGAL_NOTICE_TEMPLATE.md`
- One-click bankroll presets in the sidebar:
  - `Aggressive`
  - `Balanced`
  - `Conservative`
  - Presets auto-apply parlay and risk controls (Kelly, diversification, team-use caps, team-exposure cap, stake cap, edge threshold).
- Added `Contrarian mix (%)` control to favor a small amount of underdog-heavy or opposite-side parlay exposure for diversification against favorite-heavy slates.
- Added explicit `Opposing-side mix (%)` behavior so the parlay allocator can reserve slots for underdog-heavy / opposite-side parlays rather than only favorite-leaning combinations.
- Added a new Tennis tab powered by The Odds API tennis markets, plus optimizer support so tennis is included in maximize-profit / balanced / win-chance calculations.
- Fixed Tennis tab API errors by discovering active tennis sport keys from The Odds API `/sports` endpoint and loading whichever ATP/WTA tennis keys are currently active.
- Ownership note updated to explicitly state that Herchelle Jadhav owns the product.
- Reorganized sport-specific data loading into dedicated folders:
  - `gambling_app/sports/baseball.py`
  - `gambling_app/sports/basketball.py`
  - `gambling_app/sports/tennis.py`
  - `gambling_app/controller.py` now orchestrates sport loading for the UI.
- Added a real MLB weather adjustment to the model:
  - hot weather slightly boosts offense-friendly environments
  - strong wind out slightly increases edge
  - wind in / rain / storms slightly reduce confidence
  - impact is intentionally small and only nudges the moneyline model
- Added game-date calendar controls for exact local dates:
  - sidebar selector for `Today (YYYY-MM-DD)` and `Tomorrow (YYYY-MM-DD)`
  - date filter applies across Baseball, Basketball, and Tennis tabs
  - optimizer now scores settings against the selected day
- Added a dedicated `Simulation` tab:
  - track selected parlays from each sport tab with custom stake units
  - grade tracked parlays using final scores
  - report settled record, profit in units, and ROI for the selected date
- Added direct tracking from optimized recommendations:
  - select parlays from the bankroll allocation recommendations table
  - send selected recommended parlays straight to Simulation with editable stake
  - simulation totals can now be viewed cumulatively across all tracked dates
- Updated persistence behavior for long-term tracking:
  - saved parlays now keep full history (no dedup removal)
  - Simulation tab now shows explicit all-time record/profit/ROI metrics
- Simulation grading now records failed parlay legs and displays which pick(s) failed for settled losses.
- Simulation grading now settles parlays immediately when any completed leg fails (no longer waits for all remaining legs to finish first).
- Tennis discovery improved:
  - added region fallback (`us`, `uk`, `eu`, `au`) when fetching tennis odds
  - explicitly includes Madrid Open keys (`tennis_atp_madrid_open`, `tennis_wta_madrid_open`) in tennis loading
- Parlay selection now balances by game side across the portfolio, so if one matchup is used heavily on one side, the opposite side is favored in other parlays when available.
- Parlay building now expands the candidate pool by game: if one side of a matchup makes the top-N seed list, the opposite side is kept available for alternate parlays too.
- Balanced optimization now scores same-game opposite-side coverage, so it prefers matchups where different parlays can hold opposite sides of the same game.
- Replaced the bankroll number input with a `Money to bet ($)` slider.
- MLB pick table now shows richer game context:
  - Home/Away bullpen usage (last 3 days)
  - Home/Away lineup confirmed status
  - Home/Away lineup OPS

### Changed
- `BettingEngine` scoring now includes bullpen and lineup differentials in `power_diff`.
- Edge rows now carry additional context fields used in the UI.
- Default parlay allocation now caps repeated team exposure more tightly and warns when a team appears in multiple selected parlays.
- Added stronger concentration controls in allocation:
  - New cap: maximum total bankroll exposure per team across all selected parlays.
  - New risk metric: worst single-team exposure dollars and % of bankroll.
- `Optimize balanced` now includes concentration risk in scoring (penalizes high worst single-team exposure while balancing expected profit and 1+ win chance).

### Added
- Parlay selection checkboxes in the UI for saving individual parlays.
- Persistent parlay storage in `saved_parlays.json` for future reuse.

### Maintenance
- Introduced this `PATCH_NOTES.md` file to track every update going forward.
