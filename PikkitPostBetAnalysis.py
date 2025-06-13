## Game Time!!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from typing import Dict, List
from nba_api.stats.static import players

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load transaction data from CSV or Excel, parse dates, set index.
    """
    if file_path.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path, parse_dates=['time_placed', 'time_settled'])
    else:
        df = pd.read_csv(file_path, parse_dates=['time_placed', 'time_settled'])
    df.set_index('time_placed', inplace=True)
    return df

def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean transaction data:
      - Drop unnamed export columns
      - Localize index to UTC, convert to US/Eastern
      - Filter for NBA league
      - Drop duplicates
    """
    df = df.copy()
    df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')], inplace=True)
    df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('US/Eastern')
    return df[df['leagues'] == 'NBA'].drop_duplicates()

def load_schedule(file_path: str) -> pd.DataFrame:
    """
    Load NBA schedule from csv, combine regular & playoffs sheets
    parse dates, filter out unwanted teams.
    """
    # Load all sheets
    sheets = pd.read_excel(file_path, sheet_name=None, parse_dates=['DateTime'])
    df_list = []
    for name, sheet in sheets.items():
        # columns: 'club_name', 'DateTime'
        if 'club_name' in sheet.columns and 'DateTime' in sheet.columns:
            df_list.append(sheet[['club_name', 'DateTime']].assign(season=name))
    schedule = pd.concat(df_list, ignore_index=True)
    # Filter out specific clubs 
    schedule = schedule[~schedule['club_name'].isin(['POR', 'MIA'])]
    return schedule

def add_schedule_features(df: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Merge tip-off times from schedule into transactions,
    calculate seconds since tip-off, quarter number.
    """
    df = df.copy()
    # Prepare for merge: extract game_date and club_name from bet_info 
    df['game_date'] = df.index.date
    # merge on club name
    merged = pd.merge(
        df.reset_index(),
        schedule.rename(columns={'DateTime': 'tip_off'}),
        on=['club_name'],
        how='left'
    )
    merged.set_index('time_placed', inplace=True)

    # Calculate seconds since tip-off
    merged['seconds_since_tip'] = (merged.index - merged['tip_off']).dt.total_seconds()
    # Quarter: 1-4 based on 12-min quarters (720 seconds), else NaN or special
    merged['quarter'] = merged['seconds_since_tip'].apply(
        lambda x: int(x // 720) + 1 if x >= 0 and x < 2880 else np.nan
    )
    return merged

def classify_market_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each bet as Pre-Match or Live based on settlement vs tip-off.
    """
    df = df.copy()
    df['market_type'] = np.where(
        df['time_settled'] < df['tip_off'], 'Pre-Match', 'Live'
    )
    return df

def calculate_bet_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    sizing the bet
    """
    df = df.copy()
    df['bet_size'] = df['amount']
    return df

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features: hour, day_of_week, month, season_type.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.day_of_week + 1
    df['month'] = df.index.month
    cutoff = pd.Timestamp('2024-04-13', tz='US/Eastern')
    df['season_type'] = np.where(df.index > cutoff, 'playoff', 'regular')
    return df

# the fun part
def get_all_player_names() -> List[str]:
    """
    Retrieve active NBA player names via nba_api + extras.
    """
    api_players = players.get_players()
    active = [p['full_name'] for p in api_players if p.get('is_active')]
    extra = [
        'Dereck Lively II', 'Dereck Lively', 'Larry Nance Jr', 'Cason Wallace',
        'Zavier Simpson', 'Tosan Evbuomwan', 'Rayan Rupert', 'Marvin Bagley',
        'T.J McConnell', 'Brandon Miller', 'G. Antetokounmpo', 'Derrick Jones',
        'Amen Thompson', 'S. Gilgeous-Alexander', 'Victor Wembanyama'
    ]
    return active + extra

def extract_player_name(df: pd.DataFrame, player_names: List[str]) -> pd.DataFrame:
    """
    Parse bet_info to extract a normalized 'player_name', drop rows without a name.
    """
    df = df.copy()
    def find_name(text):
        for name in player_names:
            if name in text:
                return name.title()
        return None

    df['player_name'] = df['bet_info'].apply(find_name)
    # Normalize edge-case names
    mapping = {
        'Dereck Lively Ii': 'Dereck Lively',
        'Larry Nance Jr': 'Larry Nance Jr.',
        'T.J McConnell': 'T.J. McConnell',
        'G. Antetokounmpo': 'G. Antetokounmpo',
        'Derrick Jones': 'Derrick Jones Jr.',
        'S. Gilgeous-Alexander': 'Shai Gilgeous-Alexander'
    }
    df['player_name'] = df['player_name'].replace(mapping)
    return df[df['player_name'].notna()]

def map_prop_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map textual prop variations in bet_info to standardized 'prop_stat'.
    """
    df = df.copy()
    props = {
        'PTS+REB+AST': ['points + rebounds + assists', 'pts + reb + ast', 'total points and rebounds and assists'],
        'RBD+AST'   : ['rebounds + assists', 'reb + ast'],
        'PTS+REB'   : ['points + rebounds', 'pts + reb', 'points and rebounds'],
        'PTS+AST'   : ['points + assists', 'pts + ast', 'total points and assists'],
        '3PTM'      : ['3 pointers made', 'three pointers made', '3pt', 'threes'],
        'PTS'       : ['points', 'point'],
        'REB'       : ['rebounds', 'reb'],
        'AST'       : ['assists', 'ast'],
        'STL+BLK'   : ['steals + blocks', 'steals and blocks'],
        'BLK'       : ['blocks'],
        'STL'       : ['steals']
    }
    # greedy search 
    def find_prop(text):
        lower = text.lower() #case insensitive
        for stat, vars in props.items():
            for v in vars:
                if v in lower:
                    return stat
        return 'special(DD/TD)'
    df['prop_stat'] = df['bet_info'].apply(find_prop)
    return df

# Create stratifications-esque pivot tables incorporating dollar weighted average
def create_pivots(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create various pivot tables: sum, mean, count, weighted averages.
    """
    pivots = {}
    pivots['hourly_profit'] = pd.pivot_table(df, index='hour', values='profit', aggfunc='sum')
    pivots['book_hour_profit'] = pd.pivot_table(df, index=['sports_book','hour'], values='profit', aggfunc='mean')
    pivots['dow_trade_counts'] = pd.pivot_table(df, index='day_of_week', values='profit', aggfunc='count')
    
    # Weighted avg profit by sportsbook
    weighted = df.groupby('sports_book').apply(
        lambda g: (g['profit'] * g['amount']).sum() / g['amount'].sum()
    ).reset_index(name='weighted_avg_profit')
    pivots['weighted_avg_profit'] = weighted.set_index('sports_book')
    return pivots

def export_pivots_to_excel(pivots: Dict[str, pd.DataFrame], excel_path: str) -> None:
    """
    Export each pivot to its own sheet in an Excel workbook.
    """
    with pd.ExcelWriter(excel_path) as writer:
        for name, table in pivots.items():
            table.to_excel(writer, sheet_name=name)

def save_to_sqlite(df: pd.DataFrame, db_path: str) -> None:
    """
    Save df into a SQLite database.
    """
    conn = sqlite3.connect(db_path)
    df.reset_index().to_sql('trades', conn, if_exists='replace', index=False)
    conn.close()

def plot_pivot_table(pivot: pd.DataFrame, title: str) -> None:
    """
    Plot a bar chart for a single-column pivot table.
    """
    plt.figure(figsize=(10,5))
    pivot.plot(kind='bar', legend=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main(
    transactions_path: str = 'full.xlsx',
    schedule_path: str = 'nba_schedule.xlsx',
    excel_output: str = 'Pikkit_Pivots.xlsx',
    sqlite_db: str = 'Pikkit_Trades.db'
    
) -> None:
    # Load data
    df = load_data(transactions_path)
    df = preprocess_transactions(df)

    # Load and merge schedule
    schedule = load_schedule(schedule_path)
    df = add_schedule_features(df, schedule)
    df = classify_market_type(df)
    df = calculate_bet_size(df)

    # Time, player, prop features
    df = engineer_time_features(df)
    names = get_all_player_names()
    df = extract_player_name(df, names)
    df = map_prop_stats(df)

    # Pivot, export, save, plot
    pivots = create_pivots(df)
    export_pivots_to_excel(pivots, excel_output)
    save_to_sqlite(df, sqlite_db)
    plot_pivot_table(pivots['hourly_profit'], 'Total Profit by Hour of Day')
    plot_pivot_table(pivots['dow_trade_counts'], 'Number of Trades by Day of Week')

#sanity checks.

    # def sanity_checks(df: pd.DataFrame):
    # missing_players = df['player_name'].isna().sum()
    # missing_tips    = df['tip_off'].isna().sum()
    # print(f"→ Missing player names: {missing_players}")
    # print(f"→ Rows with no tip‑off matched: {missing_tips}")
    # assert missing_players == 0, "⚠️  Some bets failed to map to a player!"


# if __name__ == '__main__':
#     main()

