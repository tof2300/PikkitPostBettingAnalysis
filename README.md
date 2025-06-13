# Pikkit Post-Bet Data Pipeline

A clean, modular Python pipeline for loading, preprocessing, and analyzing NBA prop-bet transactions data from Pikkit.

## Features

* **Data Loading**: Supports CSV and Excel formats.
* **Preprocessing**: Drops export artifacts, normalizes timestamps (UTC → US/Eastern), filters to NBA.
* **Schedule Integration**: Merges NBA schedule to derive tip-off times, calculates seconds since tip-off and quarter.
* **Market Classification**: Flags bets as `Pre-Match` or `Live`.
* **Feature Engineering**:

  * Time-based features (hour, day of week, month, season type)
  * Player-name extraction via `nba_api` + custom mappings
  * Prop-stat mapping to standardized abbreviations
  * Bet-size calculation
* **Analysis Outputs**:

  * Pivot tables: profit by hour, mean profit by sportsbook/hour, trade counts, weighted avg profit
  * Export to Excel workbook (one sheet per pivot)
  * Save full dataset to SQLite database
  * Bar-chart visualizations for key summaries

## Requirements

* Python 3.7+
* pandas
* numpy
* matplotlib
* sqlite3 (standard library)
* nba\_api
* NBA Schedule

## Installation

```bash
git clone https://github.com/yourusername/PikkitPostBetAnalysis.git
cd pikkit-feature-engineering
pip install -r requirements.txt
```

## Usage

```bash
python PikkitPostBetAnalysis.py \
  --transactions full.xlsx \
  --schedule nba_schedule.xlsx \
  --excel-output Pikkit_Pivots.xlsx \
  --sqlite-db Pikkit_Trades.db
```

##
