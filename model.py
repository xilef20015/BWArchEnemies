import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
import requests

# Constants
API_TOKEN = '9c8a155aeb054f72a7c6b93e6416b2bf'
BASE_URL = "https://api.football-data.org/v4/competitions/BL1/matches"

# Load historical data
df_bundesliga_recencyfeatures = pd.read_csv('df_bundesliga_recencyfeatures.csv')

# Load team name mapping
team_name_mapping = {
    "1. FC Heidenheim 1846": "1. FC Heidenheim",
    "1. FC Union Berlin": "1. FC Union Berlin",
    "1. FSV Mainz 05": "1. FSV Mainz 05",
    "TSG 1899 Hoffenheim": "1899 Hoffenheim",
    "FC Bayern München": "Bayern München",
    "Borussia Mönchengladbach": "Bor. Mönchengladbach",
    "SV Werder Bremen": "Werder Bremen",
    "VfL Bochum 1848": "VfL Bochum",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "SC Freiburg": "SC Freiburg",
    "Bayer 04 Leverkusen": "Bayer 04 Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "VfB Stuttgart": "VfB Stuttgart",
    "VfL Wolfsburg": "VfL Wolfsburg",
    "FC Augsburg": "FC Augsburg",
    "FC St. Pauli 1910": "FC St. Pauli",
    "Holstein Kiel": "Holstein Kiel"
}

# Function to get upcoming matches from the API
def get_upcoming_matches():
    headers = {"X-Auth-Token": API_TOKEN}
    params = {"status": "SCHEDULED"}  # Get scheduled matches
    response = requests.get(BASE_URL, headers=headers, params=params)
    matches = response.json()['matches']
    
    # Parse and format matches
    match_list = []
    for match in matches:
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        
        # Map team names to match historical data
        home_team_mapped = team_name_mapping.get(home_team, home_team)
        away_team_mapped = team_name_mapping.get(away_team, away_team)
        
        match_list.append({
            'home_team': home_team_mapped,
            'away_team': away_team_mapped,
            'season_start_year': 2024,  # Adjust this based on the actual season
            'month': pd.to_datetime(match['utcDate']).month
        })
    
    return pd.DataFrame(match_list)

# Function to train the XGBoost model
def run_xgboost_model(df):
    features = [
        'home_recent_goals_scored_n5',  
        'home_recent_goals_conceded_n5',  
        'away_recent_goals_scored_n5',  
        'away_recent_goals_conceded_n5',  
        'home_prev_season_goals_scored',  
        'home_prev_season_goals_conceded',  
        'away_prev_season_goals_scored',  
        'away_prev_season_goals_conceded',  
        'home_prev_season_position',  
        'away_prev_season_position',  
        'home_team_pi_rating',  
        'away_team_pi_rating',
        'strength_of_opposition',  
        'strength_difference',  
        'previous_win',  
        'home_team_position',  
        'away_team_position',  
        'month'  
    ]

    X = df[features].dropna().reset_index(drop=True)
    y = df.loc[X.index, 'winner'].reset_index(drop=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost model
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Fit the model
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Log Loss: {log_loss(y_test, y_prob):.4f}")

    return model, label_encoder, scaler

# Function to prepare features for API matches
def prepare_api_match_features(df, matches_df):
    features = [
        'home_recent_goals_scored_n5',  
        'home_recent_goals_conceded_n5',  
        'away_recent_goals_scored_n5',  
        'away_recent_goals_conceded_n5',  
        'home_prev_season_goals_scored',  
        'home_prev_season_goals_conceded',  
        'away_prev_season_goals_scored',  
        'away_prev_season_goals_conceded',  
        'home_prev_season_position',  
        'away_prev_season_position',  
        'home_team_pi_rating',  
        'away_team_pi_rating',
        'strength_of_opposition',  
        'strength_difference',  
        'previous_win',  
        'home_team_position',  
        'away_team_position',  
        'month'
    ]

    feature_rows = []
    valid_matches = []  # To track valid matches
    
    for _, match in matches_df.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        
        last_season = df[df['season_start_year'] == match['season_start_year'] - 1]
        home_stats = last_season[last_season['home_team'] == home_team].iloc[-5:]
        away_stats = last_season[last_season['away_team'] == away_team].iloc[-5:]
        
        if home_stats.empty or away_stats.empty:
            print(f"Skipping match between {home_team} and {away_team} due to insufficient data.")
            continue
        
        feature_row = {
            'home_recent_goals_scored_n5': home_stats['home_goals'].sum(),
            'home_recent_goals_conceded_n5': home_stats['away_goals'].sum(),
            'away_recent_goals_scored_n5': away_stats['away_goals'].sum(),
            'away_recent_goals_conceded_n5': away_stats['home_goals'].sum(),
            'home_prev_season_goals_scored': home_stats['home_goals'].sum(),
            'home_prev_season_goals_conceded': home_stats['away_goals'].sum(),
            'away_prev_season_goals_scored': away_stats['away_goals'].sum(),
            'away_prev_season_goals_conceded': away_stats['home_goals'].sum(),
            'home_prev_season_position': home_stats['home_team_position'].iloc[-1],
            'away_prev_season_position': away_stats['away_team_position'].iloc[-1],
            'home_team_pi_rating': home_stats['home_team_pi_rating'].iloc[-1],
            'away_team_pi_rating': away_stats['away_team_pi_rating'].iloc[-1],
            'strength_of_opposition': away_stats['strength_of_opposition'].iloc[-1],
            'strength_difference': home_stats['strength_difference'].iloc[-1],
            'previous_win': home_stats['previous_win'].iloc[-1],
            'home_team_position': home_stats['home_team_position'].iloc[-1],
            'away_team_position': away_stats['away_team_position'].iloc[-1],
            'month': match['month']
        }
        
        feature_rows.append(feature_row)
        valid_matches.append(match)  # Keep track of valid matches
    
    return pd.DataFrame(feature_rows), pd.DataFrame(valid_matches)

# Function to predict outcomes based on API matches
def predict_api_outcomes(model, matches_df, scaler):
    features_df, valid_matches_df = prepare_api_match_features(df_bundesliga_recencyfeatures, matches_df)
    features_scaled = scaler.transform(features_df)
    predictions = model.predict(features_scaled)
    prediction_probabilities = model.predict_proba(features_scaled)
    
    return predictions, prediction_probabilities, valid_matches_df

# Train the model
trained_model, fitted_label_encoder, scaler = run_xgboost_model(df_bundesliga_recencyfeatures)

# Get upcoming matches from the API
upcoming_matches_df = get_upcoming_matches()

# Predict outcomes for upcoming matches
predicted_winners, predicted_probabilities, valid_matches_df = predict_api_outcomes(trained_model, upcoming_matches_df, scaler)

# Assign the predictions to the valid matches
valid_matches_df['predicted_winner'] = fitted_label_encoder.inverse_transform(predicted_winners)
valid_matches_df['home_win_prob'] = predicted_probabilities[:, 0]
valid_matches_df['draw_prob'] = predicted_probabilities[:, 1]
valid_matches_df['away_win_prob'] = predicted_probabilities[:, 2]

# Save the predictions to a CSV file
valid_matches_df.to_csv('predicted_results.csv', index=False)

print("Predicted results saved to predicted_results.csv")
