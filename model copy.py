import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to calculate recency features
def calculate_recency_features(df, n):
    df['home_recent_goals_scored'] = 0
    df['home_recent_goals_conceded'] = 0
    df['away_recent_goals_scored'] = 0
    df['away_recent_goals_conceded'] = 0
    df['home_team_position'] = np.nan
    df['away_team_position'] = np.nan

    for season in df['season_start_year'].unique():
        season_matches = df[df['season_start_year'] == season]
        teams_in_season = pd.concat([season_matches['home_team'], season_matches['away_team']]).unique()
        points_table = {team: 0 for team in teams_in_season}

        for index, row in season_matches.iterrows():
            if row['winner'] == 0:  # Home win
                points_table[row['home_team']] += 3
            elif row['winner'] == 2:  # Away win
                points_table[row['away_team']] += 3
            else:  # Draw
                points_table[row['home_team']] += 1
                points_table[row['away_team']] += 1

            # Create league table
            league_table = pd.DataFrame(list(points_table.items()), columns=['team', 'points'])
            league_table = league_table.sort_values(by='points', ascending=False).reset_index(drop=True)
            league_table['position'] = league_table.index + 1

            # Assign team positions
            if row['home_team'] in league_table['team'].values:
                df.loc[index, 'home_team_position'] = league_table[league_table['team'] == row['home_team']]['position'].values[0]
            if row['away_team'] in league_table['team'].values:
                df.loc[index, 'away_team_position'] = league_table[league_table['team'] == row['away_team']]['position'].values[0]

    # Calculate strength difference
    df['strength_difference'] = df['home_team_position'] - df['away_team_position']
    
    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    return df

# Logistic Regression Model
def run_logistic_regression(df, n):
    # Calculate recency features
    df = calculate_recency_features(df, n)
    
    # Define the feature columns
    features = ['home_recent_goals_scored', 'home_recent_goals_conceded', 
                'away_recent_goals_scored', 'away_recent_goals_conceded', 
                'home_team_position', 'away_team_position', 'strength_difference', 'month']
    
    # Prepare the feature matrix and target variable
    X = df[features]
    y = df['winner']

    # Drop rows with missing values in X or y
    X = X.dropna()
    y = y.loc[X.index]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Logistic Regression model
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"Accuracy for n={n}: {accuracy:.2f}")
    print(f"Classification Report for n={n}:\n", classification_report(y_test, y_pred))

    return model

# KNN Model
def run_knn(df, n):
    df = calculate_recency_features(df, n)
    
    features = ['home_recent_goals_scored', 'home_recent_goals_conceded', 
                'away_recent_goals_scored', 'away_recent_goals_conceded', 
                'home_team_position', 'away_team_position', 'strength_difference', 'month']
    
    X = df[features]
    y = df['winner']
    
    X = X.dropna()
    y = y.loc[X.index]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': [3, 5, 7, 10]}
    
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    best_knn_model = grid_search.best_estimator_
    
    y_pred = best_knn_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for n={n}: {accuracy:.2f}")
    print(f"Classification Report for n={n}:\n", classification_report(y_test, y_pred))
    
    return best_knn_model

# Gradient Boosting Model
def run_gradient_boosting(df, n):
    df = calculate_recency_features(df, n)
    
    features = ['home_recent_goals_scored', 'home_recent_goals_conceded', 
                'away_recent_goals_scored', 'away_recent_goals_conceded', 
                'home_team_position', 'away_team_position', 'strength_difference', 'month']
    
    X = df[features]
    y = df['winner']
    
    X = X.dropna()
    y = y.loc[X.index]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    gbdt = GradientBoostingClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'max_depth': [3],
        'subsample': [0.8]
    }
    
    grid_search = GridSearchCV(gbdt, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    best_gbdt_model = grid_search.best_estimator_
    
    y_pred = best_gbdt_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for n={n}: {accuracy:.2f}")
    print(f"Classification Report for n={n}:\n", classification_report(y_test, y_pred))
    
    return best_gbdt_model