import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests

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

# Constants for API access
API_TOKEN = '9c8a155aeb054f72a7c6b93e6416b2bf'
BASE_URL = "https://api.football-data.org/v4/competitions/BL1/matches"

# Club logos and locations
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
    "FC Bayern M\u00fcnchen": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000G.svg",
    "VfB Stuttgart": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000D.svg",
    "VfL Wolfsburg": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000003.svg",
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
    "FC Bayern M\u00fcnchen": "Allianz Arena",
    "VfB Stuttgart": "MHPArena",
    "VfL Wolfsburg": "Volkswagen Arena",
}

# Hardcoded Data (matches and predictions)
matches_data = [
    {"home_team": "Borussia Mönchengladbach", "away_team": "Bayer 04 Leverkusen", "home_win_prob": 22.79, "draw_prob": 35.23, "away_win_prob": 41.98, "total_goals": 4.9},
    {"home_team": "RB Leipzig", "away_team": "VfL Bochum 1848", "home_win_prob": 72.61, "draw_prob": 14.49, "away_win_prob": 12.90, "total_goals": 3.0},
    {"home_team": "SC Freiburg", "away_team": "VfB Stuttgart", "home_win_prob": 39.27, "draw_prob": 19.59, "away_win_prob": 41.14, "total_goals": 5.0},
    {"home_team": "FC Augsburg", "away_team": "SV Werder Bremen", "home_win_prob": 83.62, "draw_prob": 9.93, "away_win_prob": 6.44, "total_goals": 4.8},
    {"home_team": "1. FSV Mainz 05", "away_team": "1. FC Union Berlin", "home_win_prob": 32.25, "draw_prob": 37.91, "away_win_prob": 29.83, "total_goals": 3.3},
    {"home_team": "Borussia Dortmund", "away_team": "Eintracht Frankfurt", "home_win_prob": 55.27, "draw_prob": 22.43, "away_win_prob": 22.30, "total_goals": 3.5},
    {"home_team": "VfL Wolfsburg", "away_team": "FC Bayern M\u00fcnchen", "home_win_prob": 47.09, "draw_prob": 17.35, "away_win_prob": 35.56, "total_goals": 3.5}
]

# Function to fetch match dates from the API
def fetch_match_dates():
    headers = {"X-Auth-Token": API_TOKEN}
    today = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
    
    params = {
        "dateFrom": today,
        "dateTo": end_date,
        "status": "SCHEDULED",
    }
    
    response = requests.get(BASE_URL, headers=headers, params=params)
    
    if response.status_code == 200:
        matches = response.json().get('matches', [])
        match_dates = {
            (match['homeTeam']['name'], match['awayTeam']['name']): datetime.strptime(match['utcDate'], "%Y-%m-%dT%H:%M:%SZ").strftime('%d/%m/%y %H:%M')
            for match in matches
        }
        return match_dates
    else:
        st.error("Failed to fetch match dates from the API.")
        return {}

# Fetch match dates from API
match_dates = fetch_match_dates()

# Display Matches and Predictions
for match in matches_data:
    home_team = match['home_team']
    away_team = match['away_team']
    home_win_prob = match['home_win_prob']
    draw_prob = match['draw_prob']
    away_win_prob = match['away_win_prob']
    
    # Fetch the date for the match from the API
    match_key = (home_team, away_team)
    match_date = match_dates.get(match_key, "Date not available")
    home_team_logo = club_icons.get(home_team, "")
    away_team_logo = club_icons.get(away_team, "")
    location = club_locations.get(home_team, "Unknown Location")

    # Display match information (without predictions yet)
    st.markdown(f"""
        <div class="match-container">
            <div class="match-header">
                <div class="team-container">
                    <img src="{home_team_logo}" alt="{home_team} Logo" width="80" />
                    <h3 style="color: #FAFAFA;">{home_team}</h3>
                </div>
                <div class="vs-container">
                    <h2 style="color: #F9A825;">VS</h2>
                    <p style="color: #FAFAFA;">{match_date}</p>
                </div>
                <div class="team-container">
                    <img src="{away_team_logo}" alt="{away_team} Logo" width="80" />
                    <h3 style="color: #FAFAFA;">{away_team}</h3>
                </div>
            </div>
            <p class="match-location">Location: {location}</p>
        </div>
    """, unsafe_allow_html=True)

    # Show Prediction Button
    if st.button("Show Prediction", key=f"{home_team}-{away_team}"):
        st.write(f"**{home_team} Win Probability:** {home_win_prob}%")
        st.write(f"**Draw Probability:** {draw_prob}%")
        st.write(f"**{away_team} Win Probability:** {away_win_prob}%")
        st.write(f"**Total Goals Predicted:** {match['total_goals']}")


