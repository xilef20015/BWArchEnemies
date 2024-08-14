import streamlit as st
import requests
from datetime import datetime, timedelta


# Inject custom CSS for a football-themed page with background image
st.markdown(
    f"""
    <style>
    /* General page styling */
    body {{
        background-size: cover;
        font-family: 'Arial', sans-serif;
        color: #FAFAFA;
        padding: 0;
        margin: 0;
    }}
    
    /* Container for the main app */
    .main {{
        padding: 20px;
        background-color: rgba(28, 30, 38, 0.8); /* semi-transparent background */
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
    }}
    
    /* Title Styling */
    h1 {{
        text-align: center;
        color: #F9A825;
        font-size: 3rem;
        margin-bottom: 30px;
    }}

    /* Styling for the selectbox */
    .stSelectbox label {{
        font-size: 20px;
        font-weight: bold;
        color: #F9A825;
    }}
    
    .stSelectbox div[data-testid="stMarkdownContainer"] {{
        background-color: #283593;
        border-radius: 8px;
        padding: 10px;
    }}
    
    .stSelectbox div[data-testid="stMarkdownContainer"] select {{
        background-color: #1E88E5;
        color: #fff;
    }}
    
    /* Button styling */
    button {{
        background-color: #F9A825;
        color: #0B3D91;
        font-size: 18px;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }}
    
    button:hover {{
        background-color: #FFA726;
    }}
    
    /* Placeholder image styling */
    .football-image {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-top: 20px;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
        max-width: 100%;
    }}
    
    /* Prediction result styling */
    .prediction-result {{
        background-color: #283593;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-top: 30px;
        font-size: 1.5rem;
    }}

    /* Footer styling */
    footer {{
        text-align: center;
        padding: 20px;
        background-color: rgba(28, 30, 38, 0.8);
        color: #FAFAFA;
        font-size: 0.8rem;
        margin-top: 40px;
        border-radius: 15px;
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
    
    # Convert UTC date to dd/mm/yyyy format
    for match in match_list:
        match['date'] = datetime.strptime(match['date'], "%Y-%m-%dT%H:%M:%SZ").strftime('%d/%m/%Y %H:%M')
    
    return match_list


matches = get_upcoming_matches()

if matches:
    selected_match = st.selectbox(
        "Select a match:",
        matches,
        format_func=lambda match: f"{match['home_team']} vs {match['away_team']} on {match['date']}"
    )
    
    if st.button("Predict"):
        home_team = selected_match['home_team']
        away_team = selected_match['away_team']
        
        #placeholder
        st.write(f"Prediction: {home_team} vs {away_team} will end in a draw!")
else:
    st.write("No upcoming matches found.")
