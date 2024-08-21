"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import plotly.express as px




# Load the CSV file into a DataFrame
df_bundesliga_recencyfeatures = pd.read_csv('df_bundesliga_recencyfeatures.csv')

first_matchday_data = {
    'home_team': ['Bor. Mönchengladbach', 'RB Leipzig', 'TSG 1899 Hoffenheim',
                  'SC Freiburg', 'FC Augsburg', '1. FSV Mainz 05', 
                  'Borussia Dortmund', 'VfL Wolfsburg', 'FC St. Pauli'],  # Home teams
    'away_team': ['Bayer 04 Leverkusen', 'VfL Bochum', 'Holstein Kiel',
                  'VfB Stuttgart', 'Werder Bremen', '1. FC Union Berlin', 
                  'Eintracht Frankfurt', 'Bayern München', '1. FC Heidenheim'],  # Away teams
    'season_start_year': [2024] * 9,
    'month': [8] * 9
}

# Convert to DataFrame
first_matchday_df = pd.DataFrame(first_matchday_data)


def run_gradient_boosting(df, n):

    features = [
        'home_recent_goals_scored_n10', 
        'home_recent_goals_conceded_n10',
        'away_recent_goals_scored_n10',  
        'away_recent_goals_conceded_n10',
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

    X = df[features]
    y = df['winner'] 

    X = X.dropna()
    y = y.loc[X.index] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # GBDT model 
    gbdt = GradientBoostingClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'max_depth': [3],
        'subsample': [0.8],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }

    grid_search = GridSearchCV(gbdt, param_grid, cv=5, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Best Parameters for n={n}: {best_params}")

    best_gbdt_model = grid_search.best_estimator_

    y_pred = best_gbdt_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for n={n}: {accuracy:.2f}")

    print(f"Classification Report for n={n}:\n", classification_report(y_test, y_pred))

    cv_scores = cross_val_score(best_gbdt_model, X_train, y_train, cv=5)
    print(f"Cross-validated accuracy for n={n}: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    return best_gbdt_model
gbdt_model_n9 = run_gradient_boosting(df_bundesliga_recencyfeatures.copy(), 10)

# Calculate RPS Score 

def calculate_rps(y_true, y_prob):
    rps = 0.0
    num_classes = y_prob.shape[1]  # Number of classes
    for i in range(len(y_true)):
        true_class = int(y_true[i])
        
        # Ensure true_class is a valid index
        if true_class >= num_classes:
            raise ValueError(f"True class index {true_class} is out of range for the number of classes {num_classes}")

        # Cumulative probabilities
        cumulative_prob = np.cumsum(y_prob[i])

        # Cumulative true distribution
        cumulative_true = np.zeros(num_classes)
        cumulative_true[true_class] = 1
        cumulative_true = np.cumsum(cumulative_true)

        # Calculate RPS for this instance
        rps += np.sum((cumulative_prob - cumulative_true) ** 2) / (num_classes - 1)

    return rps / len(y_true)


def run_xgboost_model_reduced(df, n):


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

    if not all(feature in df.columns for feature in features):
        raise ValueError(f"One or more of the required features are missing in the DataFrame. Available columns are: {df.columns.tolist()}")

    X = df[features]
    y = df['winner']

    X = X.dropna()
    y = y.loc[X.index] 

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    class_names = [f"Class {i}" for i in range(len(np.unique(y)))]

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
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    log_loss_value = log_loss(y_test, y_prob)
    rps_value = calculate_rps(y_test, y_prob)
    
    print(f"Accuracy for n={n}: {accuracy:.2f}")
    print(f"Log Loss for n={n}: {log_loss_value:.4f}")
    print(f"Ranked Probability Score (RPS) for n={n}: {rps_value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return model, label_encoder

model = run_xgboost_model_reduced(df_bundesliga_recencyfeatures, n=10)


# Predict Winner for next season

# Function to prepare the first matchday features
def prepare_first_matchday_features(df, first_matchday):
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
    
    first_matchday_features = pd.DataFrame(columns=features)
    valid_matches = []  # To keep track of matches with valid data
    
    for _, match in first_matchday.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        
        last_season = df[df['season_start_year'] == match['season_start_year'] - 1]
        
        if last_season[last_season['home_team'] == home_team].empty or last_season[last_season['away_team'] == away_team].empty:
            print(f"Data missing for either {home_team} or {away_team} in the previous season. Skipping match.")
            continue


        #Extract last 5 matches 
        home_stats = last_season[last_season['home_team'] == home_team].iloc[-5:]
        away_stats = last_season[last_season['away_team'] == away_team].iloc[-5:]
        
        if home_stats.empty or away_stats.empty:
            print(f"Insufficient data for {home_team} or {away_team} in the last 5 games of the previous season.")
            continue
        
        feature_row = pd.DataFrame([{
            'home_recent_goals_scored_n5': home_stats['home_goals'].sum(),
            'home_recent_goals_conceded_n5': home_stats['away_goals'].sum(),
            'away_recent_goals_scored_n5': away_stats['away_goals'].sum(),
            'away_recent_goals_conceded_n5': away_stats['home_goals'].sum(),
            'home_prev_season_goals_scored': home_stats['home_goals'].sum(),
            'home_prev_season_goals_conceded': home_stats['away_goals'].sum(),
            'away_prev_season_goals_scored': away_stats['away_goals'].sum(),
            'away_prev_season_goals_conceded': away_stats['home_goals'].sum(),
            'home_prev_season_position': last_season[last_season['home_team'] == home_team]['home_team_position'].iloc[-1] if not last_season[last_season['home_team'] == home_team].empty else np.nan,
            'away_prev_season_position': last_season[last_season['away_team'] == away_team]['away_team_position'].iloc[-1] if not last_season[last_season['away_team'] == away_team].empty else np.nan,
            'home_team_pi_rating': last_season[last_season['home_team'] == home_team]['home_team_pi_rating'].iloc[-1] if not last_season[last_season['home_team'] == home_team].empty else np.nan,
            'away_team_pi_rating': last_season[last_season['away_team'] == away_team]['away_team_pi_rating'].iloc[-1] if not last_season[last_season['away_team'] == away_team].empty else np.nan,
            'strength_of_opposition': last_season[last_season['away_team'] == away_team]['strength_of_opposition'].iloc[-1] if not last_season[last_season['away_team'] == away_team].empty else np.nan,
            'strength_difference': last_season[last_season['home_team'] == home_team]['strength_difference'].iloc[-1] if not last_season[last_season['home_team'] == home_team].empty else np.nan,
            'previous_win': last_season[last_season['home_team'] == home_team]['previous_win'].iloc[-1] if not last_season[last_season['home_team'] == home_team].empty else np.nan,
            'home_team_position': last_season[last_season['home_team'] == home_team]['home_team_position'].iloc[-1] if not last_season[last_season['home_team'] == home_team].empty else np.nan,
            'away_team_position': last_season[last_season['away_team'] == away_team]['away_team_position'].iloc[-1] if not last_season[last_season['away_team'] == away_team].empty else np.nan,
            'month': match['month']
        }])
        
        first_matchday_features = pd.concat([first_matchday_features, feature_row], ignore_index=True)
        valid_matches.append(match)
    
    return first_matchday_features, pd.DataFrame(valid_matches)
first_matchday_data = {
    'home_team': ['Bor. Mönchengladbach', 'RB Leipzig', 'TSG 1899 Hoffenheim',
                  'SC Freiburg', 'FC Augsburg', '1. FSV Mainz 05', 
                  'Borussia Dortmund', 'VfL Wolfsburg', 'FC St. Pauli'],  # Home teams
    'away_team': ['Bayer 04 Leverkusen', 'VfL Bochum', 'Holstein Kiel',
                  'VfB Stuttgart', 'Werder Bremen', '1. FC Union Berlin', 
                  'Eintracht Frankfurt', 'Bayern München', '1. FC Heidenheim'],  # Away teams
    'season_start_year': [2024] * 9,
    'month': [8] * 9
}

# Convert to DataFrame
first_matchday_df = pd.DataFrame(first_matchday_data)
# Predict outcomes with probabilities
def predict_first_matchday_outcomes_with_probabilities(model, first_matchday_features):
 
    scaler = StandardScaler()
    first_matchday_features_scaled = scaler.fit_transform(first_matchday_features)
    

    predictions = model.predict(first_matchday_features_scaled)
    prediction_probabilities = model.predict_proba(first_matchday_features_scaled)
    
    return predictions, prediction_probabilities





trained_model, fitted_label_encoder = run_xgboost_model_reduced(df_bundesliga_recencyfeatures, n=10)

# Prepare the first matchday data and get valid matches
first_matchday_features, valid_first_matchday_df = prepare_first_matchday_features(df_bundesliga_recencyfeatures, first_matchday_df)

# Predict outcomes for the valid matches
predicted_winners, predicted_probabilities = predict_first_matchday_outcomes_with_probabilities(trained_model, first_matchday_features)

valid_first_matchday_df['predicted_winner'] = predicted_winners
valid_first_matchday_df['home_win_prob'] = predicted_probabilities[:, 0]
valid_first_matchday_df['draw_prob'] = predicted_probabilities[:, 1]
valid_first_matchday_df['away_win_prob'] = predicted_probabilities[:, 2]

valid_first_matchday_df
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the CSV file into a DataFrame
df_bundesliga_recencyfeatures = pd.read_csv('df_bundesliga_recencyfeatures.csv')

# First matchday data
first_matchday_data = {
    'home_team': ['Bor. Mönchengladbach', 'RB Leipzig', 'TSG 1899 Hoffenheim',
                  'SC Freiburg', 'FC Augsburg', '1. FSV Mainz 05', 
                  'Borussia Dortmund', 'VfL Wolfsburg', 'FC St. Pauli'],  # Home teams
    'away_team': ['Bayer 04 Leverkusen', 'VfL Bochum', 'Holstein Kiel',
                  'VfB Stuttgart', 'Werder Bremen', '1. FC Union Berlin', 
                  'Eintracht Frankfurt', 'Bayern München', '1. FC Heidenheim'],  # Away teams
    'season_start_year': [2024] * 9,
    'month': [8] * 9
}
first_matchday_df = pd.DataFrame(first_matchday_data)

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

    # Convert class names to strings to avoid the TypeError
    class_names = [str(c) for c in label_encoder.classes_]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return model, label_encoder, scaler

# Function to prepare first matchday features
def prepare_first_matchday_features(df, first_matchday_df):
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
    valid_matches = []  # This will keep track of matches with valid data
    
    for _, match in first_matchday_df.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        season_year = match['season_start_year'] - 1
        
        last_season = df[df['season_start_year'] == season_year]
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
        valid_matches.append(match)  # Add valid match to list
    
    first_matchday_features = pd.DataFrame(feature_rows)
    valid_first_matchday_df = pd.DataFrame(valid_matches)  # Convert valid matches to DataFrame
    
    return first_matchday_features, valid_first_matchday_df

# Predict outcomes for valid matches only
def predict_first_matchday_outcomes(model, first_matchday_features, scaler):
    first_matchday_features_scaled = scaler.transform(first_matchday_features)
    predictions = model.predict(first_matchday_features_scaled)
    prediction_probabilities = model.predict_proba(first_matchday_features_scaled)
    
    return predictions, prediction_probabilities

# Train the model
trained_model, fitted_label_encoder, scaler = run_xgboost_model(df_bundesliga_recencyfeatures)

# Prepare first matchday data and get valid matches
first_matchday_features, valid_first_matchday_df = prepare_first_matchday_features(df_bundesliga_recencyfeatures, first_matchday_df)

# Predict outcomes for the valid matches
predicted_winners, predicted_probabilities = predict_first_matchday_outcomes(trained_model, first_matchday_features, scaler)

# Assign the predictions only to the valid matches
valid_first_matchday_df['predicted_winner'] = fitted_label_encoder.inverse_transform(predicted_winners)
valid_first_matchday_df['home_win_prob'] = predicted_probabilities[:, 0]
valid_first_matchday_df['draw_prob'] = predicted_probabilities[:, 1]
valid_first_matchday_df['away_win_prob'] = predicted_probabilities[:, 2]

# Print the predictions
print(valid_first_matchday_df[['home_team', 'away_team', 'predicted_winner', 'home_win_prob', 'draw_prob', 'away_win_prob']])

