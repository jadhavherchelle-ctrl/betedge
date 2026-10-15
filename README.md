# Herchelles ⚾ MLB Live Odds + Best Bet/Parlay Finder

Simple Streamlit app that:
- pulls **live MLB moneyline odds** from The Odds API,
- pulls **MLB standings stats** from MLB Stats API,
- computes a transparent **edge model** for each side,
- shows top straight-bet candidates,
- generates 2/3/4-leg parlay ideas from top picks.
- includes a **Basketball** tab with live NBA odds and ESPN standings-based team strength.

## Project structure
- [app.py](app.py): Streamlit launcher
- [gambling_app/controller.py](gambling_app/controller.py): sport loading orchestration
- [gambling_app/sports/](gambling_app/sports): sport-specific loaders and helpers
- [gambling_app/services.py](gambling_app/services.py): API and stats fetchers
- [gambling_app/engine.py](gambling_app/engine.py): odds model, parlays, bankroll logic
- [gambling_app/ui.py](gambling_app/ui.py): Streamlit layout and tabs
- [gambling_app/constants.py](gambling_app/constants.py): sport/team constants

## 1) Get an API key
Create a key at: https://the-odds-api.com

## 2) Install
```bash
pip install -r requirements.txt
```

## 3) Set key (optional)
```bash
export ODDS_API_KEY="YOUR_KEY_HERE"
```

## 4) Run
```bash
streamlit run app.py
```

## 5) Replay yesterday (MLB simulation)
You can run a quick "would this have worked yesterday?" replay using historical MLB odds + final scores.

```bash
python simulate_yesterday.py --min-edge 2.0 --home-field-adv 0.06
```

Requirements:
- `ODDS_API_KEY` must be set.
- Your Odds API plan must allow historical odds endpoint access.

The replay assumes **1 unit per qualifying pick** and prints win/loss, units won/lost, and ROI for yesterday.

## Notes
- This is an educational model, not betting or financial advice.
- The model is intentionally simple (win% + run differential + home field factor).
- You can tune edge threshold, season, and parlay settings in the sidebar.
