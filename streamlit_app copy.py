import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import requests
from model import run_logistic_regression, calculate_recency_features

# Caching data load using st.cache_data
@st.cache_data
def load_data():
    return pd.read_csv('df_bundesliga_recencyfeatures.csv')

df_bundesliga = load_data()

# Caching model load using st.cache_resource
@st.cache_resource
def load_classification_model(df, n=5):
    model = run_logistic_regression(df, n)
    # Fit the scaler only once during model training
    scaler = StandardScaler()
    df_temp = df[['home_recent_goals_scored', 'home_recent_goals_conceded', 
                  'away_recent_goals_scored', 'away_recent_goals_conceded', 
                  'home_team_position', 'away_team_position', 'strength_difference', 'month']].dropna()
    scaler.fit(df_temp)
    return model, scaler

model, scaler = load_classification_model(df_bundesliga.copy(), n=5)

# Inject custom CSS for styling
st.markdown(
    f"""
    <style>
    body {{
        background-size: cover;
        font-family: 'Arial', sans-serif;
        color: #FAFAFA;
        padding: 0;
        margin: 0;
    }}
    
    .main {{
        padding: 20px;
        background-color: rgba(28, 30, 38, 0.8);
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
    }}
    
    h1 {{
        text-align: center;
        color: #F9A825;
        font-size: 3rem;
        margin-bottom: 30px;
    }}

    .match-container {{
        background-color: #283593;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
    }}
    
    .match-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        text-align: center;
    }}

    .team-container {{
        flex: 1;
        text-align: center;
    }}

    .vs-container {{
        flex: 0.2;
        text-align: center;
    }}

    .match-location {{
        text-align: center;
        color: #FAFAFA;
        margin-top: 10px;
    }}

    .predict-button-container {{
        text-align: center;
        margin-top: 20px;
    }}

    .stButton>button {{
        background-color: #F9A825;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }}
    
    .stButton>button:hover {{
        background-color: #FFA726;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app layout
st.markdown(f'<img src="https://static.vecteezy.com/system/resources/previews/010/994/314/non_2x/bundesliga-logo-symbol-with-name-design-germany-football-european-countries-football-teams-illustration-with-black-background-free-vector.jpg" class="football-image" alt="Bundesliga Logo">', unsafe_allow_html=True)
st.markdown("<h1>Bundesliga Match Prediction</h1>", unsafe_allow_html=True)

# Constants
API_TOKEN = '9c8a155aeb054f72a7c6b93e6416b2bf'
BASE_URL = "https://api.football-data.org/v4/competitions/BL1/matches"

# Create a mapping between API team names and model team names
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

# Function to map API team names to model team names
def map_team_name(api_team_name):
    return team_name_mapping.get(api_team_name, api_team_name)

# Example of usage
api_team = "1. FC Heidenheim 1846"
model_team = map_team_name(api_team)
print(f"API Team: {api_team}, Mapped to Model Team: {model_team}")


club_icons = {
    "FC Augsburg": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000010.svg",
    "1. FC Union Berlin": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000V.svg",
    "VfL Bochum 1848": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000S.svg",
    "SV Werder Bremen": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000E.svg",
    "Borussia Dortmund": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000007.svg",
    "Eintracht Frankfurt": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000F.svg",
    "SC Freiburg": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000A.svg",
    "1. FC Heidenheim 1846": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000018.svg",
    "TSG 1899 Hoffenheim": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000002.svg",
    "RB Leipzig": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000017.svg",
    "Bayer 04 Leverkusen": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000B.svg",
    "1. FSV Mainz 05": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000006.svg",
    "Borussia Mönchengladbach": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000004.svg",
    "FC Bayern München": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000G.svg",
    "VfB Stuttgart": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000D.svg",
    "VfL Wolfsburg": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000003.svg",
    "FC St. Pauli 1910": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000H.svg",
    "Holstein Kiel": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000N5P.svg"
}


club_locations = {
    "FC Augsburg": "WWK ARENA",
    "1. FC Union Berlin": "An der Alten Försterei",
    "VfL Bochum 1848": "Vonovia Ruhrstadion",
    "SV Werder Bremen": "Weser-Stadion",
    "Borussia Dortmund": "SIGNAL IDUNA PARK",
    "Eintracht Frankfurt": "Deutsche Bank Park",
    "SC Freiburg": "Europa-Park Stadion",
    "1. FC Heidenheim 1846": "Voith-Arena",
    "TSG 1899 Hoffenheim": "PreZero Arena",
    "RB Leipzig": "Red Bull Arena",
    "Bayer 04 Leverkusen": "BayArena",
    "1. FSV Mainz 05": "MEWA ARENA",
    "Borussia Mönchengladbach": "BORUSSIA-PARK",
    "FC Bayern München": "Allianz Arena",
    "VfB Stuttgart": "MHPArena",
    "VfL Wolfsburg": "Volkswagen Arena",
    "FC St. Pauli": "Millerntor-Stadion",
    "Holstein Kiel": "Holstein-Stadion"
}


# Function to fetch matches
def get_upcoming_matches():
    today = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')

    headers = {"X-Auth-Token": API_TOKEN}
    params = {
        "dateFrom": today,
        "dateTo": end_date,
        "status": "SCHEDULED",
    }

    response = requests.get(BASE_URL, headers=headers, params=params)
    matches = response.json()

    match_list = [
        {
            "home_team": match['homeTeam']['name'],
            "away_team": match['awayTeam']['name'],
            "date": match['utcDate']
        }
        for match in matches['matches']
    ]

    # Convert UTC date to dd/mm/yy format
    for match in match_list:
        match['date'] = datetime.strptime(match['date'], "%Y-%m-%dT%H:%M:%SZ").strftime('%d/%m/%y %H:%M')


    return match_list

# Function to predict the outcome using the classification model
def get_match_classification_prediction(home_team, away_team, model, scaler):
    # Map the team names to the model's names
    home_team_mapped = map_team_name(home_team)
    away_team_mapped = map_team_name(away_team)

    # Filter the DataFrame for the relevant matches for these two teams
    df_temp = df_bundesliga[
        ((df_bundesliga['home_team'] == home_team_mapped) & (df_bundesliga['away_team'] == away_team_mapped)) |
        ((df_bundesliga['home_team'] == away_team_mapped) & (df_bundesliga['away_team'] == home_team_mapped))
    ].copy()

    # Check if there is data available for the teams
    if df_temp.empty:
        return "No sufficient data to make a prediction."

    # Calculate features for both teams
    recency_features_df = calculate_recency_features(df_temp, n=5)
    feature_columns = ['home_recent_goals_scored', 'home_recent_goals_conceded', 
                       'away_recent_goals_scored', 'away_recent_goals_conceded', 
                       'home_team_position', 'away_team_position', 'strength_difference', 'month']
    
    X_input = recency_features_df[feature_columns].tail(1)
    
    # Scale input features using pre-fitted scaler
    X_scaled = scaler.transform(X_input)
    
    # Get predicted probabilities for home win, draw, and away win
    prediction_probs = model.predict_proba(X_scaled)

    # Format the output
    home_win_prob = round(prediction_probs[0][0] * 100, 2)
    draw_prob = round(prediction_probs[0][1] * 100, 2)
    away_win_prob = round(prediction_probs[0][2] * 100, 2)
    
    prediction_text = (
        f"**Prediction:**\n"
        f"- **{home_team} Win Probability:** {home_win_prob}%\n"
        f"- **Draw Probability:** {draw_prob}%\n"
        f"- **{away_team} Win Probability:** {away_win_prob}%"
    )
    
    return prediction_text

matches = get_upcoming_matches()

if matches:
    for match in matches:
        home_team = match['home_team']
        away_team = match['away_team']
        date = match['date']
        
        home_team_logo = club_icons.get(home_team, "")
        away_team_logo = club_icons.get(away_team, "")
        location = club_locations.get(home_team, "Unknown Location")

        st.markdown(f"""
            <div class="match-container">
                <div class="match-header">
                    <div class="team-container">
                        <img src="{home_team_logo}" alt="{home_team} Logo" width="80" />
                        <h3 style="color: #FAFAFA;">{home_team}</h3>
                    </div>
                    <div class="vs-container">
                        <h2 style="color: #F9A825;">VS</h2>
                        <p style="color: #FAFAFA;">{date}</p>
                    </div>
                    <div class="team-container">
                        <img src="{away_team_logo}" alt="{away_team} Logo" width="80" />
                        <h3 style="color: #FAFAFA;">{away_team}</h3>
                    </div>
                </div>
                <p class="match-location">Location: {location}</p>
            </div>
        """, unsafe_allow_html=True)

        # Functional yellow button with styling
        if st.button(f"Predict Results", key=f"predict-{home_team}-{away_team}"):
            prediction = get_match_classification_prediction(home_team, away_team, model, scaler)
            st.write(prediction)

else:
    st.write("No upcoming matches found.")
