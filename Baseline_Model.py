import pandas as pd
import numpy as np
import seaborn as sns

from Metrics import *

from sklearn.metrics import accuracy_score, confusion_matrix, root_mean_squared_error, make_scorer

df_data = pd.read_csv("/Users/nilsweigeldt/Desktop/capstone2/BWArchenemies/data/bundesliga.csv")
y_target = df_data.winner.values

"""
    Baseline model for Win Probability Prediction:
    relative frequncy of possible outcomes 
"""


win_prob = df_data.winner.value_counts(normalize=True)
win_prob = win_prob.values

y_pred = np.ones((len(df_data), len(win_prob))) * win_prob
print(y_pred)
    
avg_RPS = make_scorer(avg_ranked_probability_score, needs_proba=True, greater_is_better=False, response_methode=["predict_proba", "predict"])
#avg_rps = avg_RPS(y_true=y_target, y_pred_proba=y_pred)
avg_rps = avg_ranked_probability_score(y_target, y_pred)
print(avg_rps)


"""
    Baseline Model for exact score prediction
    Mean/Median Scored goals for home and visitors
"""

y_target_hg = df_data.home_goals.values
y_target_ag = df_data.away_goals.values
y_target_diff = y_target_hg - y_target_ag

home_score_mean = np.mean(y_target_hg)
away_score_mean = np.mean(y_target_ag)

print(home_score_mean, away_score_mean)

y_pred_hg = np.ones(len(y_target_hg)) * home_score_mean
y_pred_ag = np.ones(len(y_target_ag)) * away_score_mean
y_pred_diff = y_pred_hg - y_pred_ag

rmse_hg = root_mean_squared_error(y_pred_hg, y_target_hg)
rmse_ag = root_mean_squared_error(y_pred_ag, y_target_ag)
rmse_diff = root_mean_squared_error(y_pred_diff, y_target_diff)
print(rmse_hg, rmse_ag, rmse_diff)