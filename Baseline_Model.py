import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from Metrics import *

data_path = "bundesliga.csv"

class BaselineModel:
    def __init__(self, data_path):
        """
        Initialize the BaselineModel with the data from the CSV file.
        
        Args:
        data_path (str): The path to the CSV file containing match data.
        """
        # Load dataset
        self.df_data = pd.read_csv(data_path)

        # Calculate team-specific averages for home and away performance
        self.team_stats = self.calculate_team_stats()

    def calculate_team_stats(self):
        """
        Calculate team-specific statistics for both home and away games.
        
        Returns:
        pandas.DataFrame: A DataFrame containing the average goals scored and conceded 
        for each team at home and away.
        """
        # Group by home team and away team to calculate team-specific stats
        home_stats = self.df_data.groupby('home_team').agg(
            home_goals_mean=('home_goals', 'mean'),
            home_goals_conceded_mean=('away_goals', 'mean')
        )
        
        away_stats = self.df_data.groupby('away_team').agg(
            away_goals_mean=('away_goals', 'mean'),
            away_goals_conceded_mean=('home_goals', 'mean')
        )

        # Merge both stats to create a full profile for each team
        team_stats = home_stats.join(away_stats, how='outer')

        return team_stats

    def predict(self, home_team, away_team):
        """
        Predict the outcome of a match between the home and away teams.
        
        Args:
        home_team (str): The name of the home team.
        away_team (str): The name of the away team.
        
        Returns:
        dict: A dictionary containing predicted home goals, away goals.
        """
        # Retrieve the statistics for both teams
        if home_team in self.team_stats.index and away_team in self.team_stats.index:
            home_goals = self.team_stats.at[home_team, 'home_goals_mean']
            home_conceded = self.team_stats.at[home_team, 'home_goals_conceded_mean']
            away_goals = self.team_stats.at[away_team, 'away_goals_mean']
            away_conceded = self.team_stats.at[away_team, 'away_goals_conceded_mean']

            # Predict goals based on the average goals scored and conceded
            predicted_home_goals = (home_goals + away_conceded) / 2
            predicted_away_goals = (away_goals + home_conceded) / 2

            prediction = {
                "home_goals": predicted_home_goals,
                "away_goals": predicted_away_goals
            }
        else:
            # If teams not found in data, fallback to global averages
            predicted_home_goals = self.df_data['home_goals'].mean()
            predicted_away_goals = self.df_data['away_goals'].mean()

            prediction = {
                "home_goals": predicted_home_goals,
                "away_goals": predicted_away_goals
            }

        return prediction

    def evaluate_rmse(self):
        """
        Evaluate the RMSE for home goals, away goals, and goal differences.

        Returns:
        tuple: RMSE for home goals, away goals, and goal difference.
        """
        y_target_hg = self.df_data['home_goals'].values
        y_target_ag = self.df_data['away_goals'].values
        y_target_diff = y_target_hg - y_target_ag

        y_pred_hg = np.ones(len(y_target_hg)) * self.df_data['home_goals'].mean()
        y_pred_ag = np.ones(len(y_target_ag)) * self.df_data['away_goals'].mean()
        y_pred_diff = y_pred_hg - y_pred_ag

        rmse_hg = np.sqrt(mean_squared_error(y_pred_hg, y_target_hg))
        rmse_ag = np.sqrt(mean_squared_error(y_pred_ag, y_target_ag))
        rmse_diff = np.sqrt(mean_squared_error(y_pred_diff, y_target_diff))

        return rmse_hg, rmse_ag, rmse_diff