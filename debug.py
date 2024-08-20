import pandas as pd
import requests

# Constants
API_TOKEN = '9c8a155aeb054f72a7c6b93e6416b2bf'
BASE_URL = "https://api.football-data.org/v4/competitions/BL1/matches"

# Function to get team names from the API
def get_team_names_from_api():
    headers = {"X-Auth-Token": API_TOKEN}
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    end_date = (pd.Timestamp.now() + pd.Timedelta(days=14)).strftime('%Y-%m-%d')

    params = {
        "dateFrom": today,
        "dateTo": end_date,
        "status": "SCHEDULED",
    }

    response = requests.get(BASE_URL, headers=headers, params=params)
    matches = response.json()

    teams_api = set()

    for match in matches['matches']:
        teams_api.add(match['homeTeam']['name'])
        teams_api.add(match['awayTeam']['name'])

    return sorted(teams_api)

# Function to get team names from the model dataset
def get_team_names_from_model(dataset_path):
    df = pd.read_csv(dataset_path)
    teams_model = set(df['home_team'].unique()).union(set(df['away_team'].unique()))
    
    return sorted(teams_model)

if __name__ == "__main__":
    # Path to your CSV file with model data
    dataset_path = 'df_bundesliga_recencyfeatures.csv'
    
    print("Fetching team names from API...")
    teams_from_api = get_team_names_from_api()
    print("Teams from API:")
    for team in teams_from_api:
        print(team)
    
    print("\nFetching team names from model dataset...")
    teams_from_model = get_team_names_from_model(dataset_path)
    print("Teams from Model Dataset:")
    for team in teams_from_model:
        print(team)
