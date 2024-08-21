import pandas as pd
import requests

# Constants
API_TOKEN = '9c8a155aeb054f72a7c6b93e6416b2bf'
BASE_URL = "https://api.football-data.org/v4/competitions/BL1/matches"

# Load the historical data from CSV
def load_data():
    return pd.read_csv('df_bundesliga_recencyfeatures.csv')

df_bundesliga = load_data()

# Get the unique team names from the historical data
historical_teams_home = df_bundesliga['home_team'].unique()
historical_teams_away = df_bundesliga['away_team'].unique()
historical_teams = set(historical_teams_home).union(set(historical_teams_away))

# Fetch upcoming matches from the API
def fetch_upcoming_matches():
    headers = {"X-Auth-Token": API_TOKEN}
    response = requests.get(BASE_URL, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from API. Status code: {response.status_code}")
    
    matches = response.json()['matches']
    return matches

# Extract team names from the API matches
def get_api_team_names(matches):
    api_teams = set()
    for match in matches:
        api_teams.add(match['homeTeam']['name'])
        api_teams.add(match['awayTeam']['name'])
    return api_teams

# Compare team names from API and historical data
def compare_team_names():
    # Fetch data from API
    matches = fetch_upcoming_matches()
    
    # Get team names from API
    api_teams = get_api_team_names(matches)
    
    # Compare historical team names with API team names
    missing_in_api = historical_teams - api_teams
    missing_in_historical = api_teams - historical_teams
    
    # Print the differences
    print("\nTeams in Historical Data but missing in API Data:")
    print(missing_in_api)
    
    print("\nTeams in API Data but missing in Historical Data:")
    print(missing_in_historical)

if __name__ == "__main__":
    compare_team_names()
