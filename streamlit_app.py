import streamlit as st
import requests
from datetime import datetime, timedelta

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

# Club icons mapping
club_icons = {
    "FC Augsburg": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000010.svg",
    "1. FC Union Berlin": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000V.svg",
    "VfL Bochum 1848": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000S.svg",
    "SV Werder Bremen": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000E.svg",
    "FC St. Pauli": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000H.svg", 
    "Borussia Dortmund": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000007.svg",
    "Eintracht Frankfurt": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000F.svg",
    "SC Freiburg": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000A.svg",
    "1. FC Heidenheim 1846": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000018.svg",
    "TSG Hoffenheim": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000002.svg",
    "Holstein Kiel": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000N5P.svg",
    "RB Leipzig": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000017.svg",
    "Bayer 04 Leverkusen": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000B.svg",
    "1. FSV Mainz 05": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000006.svg",
    "Borussia Mönchengladbach": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000004.svg",
    "FC Bayern München": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000G.svg",
    "VfB Stuttgart": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-00000D.svg",
    "VfL Wolfsburg": "https://www.bundesliga.com/assets/clublogo/DFL-CLU-000003.svg"
}

# Club locations mapping
club_locations = {
    "FC Augsburg": "WWK ARENA",
    "1. FC Union Berlin": "An der Alten Försterei",
    "VfL Bochum 1848": "Vonovia Ruhrstadion",
    "SV Werder Bremen": "Weser-Stadion",
    "FC St. Pauli": "Millerntor-Stadion",
    "Borussia Dortmund": "SIGNAL IDUNA PARK",
    "Eintracht Frankfurt": "Deutsche Bank Park",
    "SC Freiburg": "Europa-Park Stadion",
    "1. FC Heidenheim 1846": "Voith-Arena",
    "TSG Hoffenheim": "PreZero Arena",
    "Holstein Kiel": "Holstein-Stadion",
    "RB Leipzig": "Red Bull Arena",
    "Bayer 04 Leverkusen": "BayArena",
    "1. FSV Mainz 05": "MEWA ARENA",
    "Borussia Mönchengladbach": "BORUSSIA-PARK",
    "FC Bayern München": "Allianz Arena",
    "VfB Stuttgart": "MHPArena",
    "VfL Wolfsburg": "Volkswagen Arena"
}

# Name mapping from API to desired display names
name_mapping = {
    "FC Augsburg": "FC Augsburg",
    "1. FC Union Berlin": "1. FC Union Berlin",
    "VfL Bochum 1848": "VfL Bochum 1848",
    "SV Werder Bremen": "SV Werder Bremen",
    "FC St. Pauli 1910": "FC St. Pauli",
    "Borussia Dortmund": "Borussia Dortmund",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "SC Freiburg": "SC Freiburg",
    "1. FC Heidenheim 1846": "1. FC Heidenheim 1846",
    "TSG 1899 Hoffenheim": "TSG Hoffenheim",
    "Holstein Kiel": "Holstein Kiel",
    "RB Leipzig": "RB Leipzig",
    "Bayer 04 Leverkusen": "Bayer 04 Leverkusen",
    "1. FSV Mainz 05": "1. FSV Mainz 05",
    "Borussia Mönchengladbach": "Borussia Mönchengladbach",
    "FC Bayern München": "FC Bayern München",
    "VfB Stuttgart": "VfB Stuttgart",
    "VfL Wolfsburg": "VfL Wolfsburg"
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
    
    # Convert UTC date to dd/mm/yyyy format
    for match in match_list:
        match['date'] = datetime.strptime(match['date'], "%Y-%m-%dT%H:%M:%SZ").strftime('%d/%m/%Y %H:%M')
    
    return match_list


matches = get_upcoming_matches()

if matches:
    for match in matches:
        home_team = name_mapping.get(match['home_team'], match['home_team'])
        away_team = name_mapping.get(match['away_team'], match['away_team'])
        date = match['date']
        location = club_locations.get(home_team, "Unknown Location")
        
        st.markdown(f"""
            <div class="match-container">
                <div class="match-header">
                    <div class="team-container">
                        <img src="{club_icons[home_team]}" alt="{home_team} Logo" width="80" />
                        <h3 style="color: #FAFAFA;">{home_team}</h3>
                    </div>
                    <div class="vs-container">
                        <h2 style="color: #F9A825;">VS</h2>
                        <p style="color: #FAFAFA;">{date}</p>
                    </div>
                    <div class="team-container">
                        <img src="{club_icons[away_team]}" alt="{away_team} Logo" width="80" />
                        <h3 style="color: #FAFAFA;">{away_team}</h3>
                    </div>
                </div>
                <p class="match-location">Location: {location}</p>
                <div class="predict-button-container">
        """, unsafe_allow_html=True)
        
        if st.button(f"Predict {home_team} vs {away_team}"):
            st.write(f"Prediction: {home_team} vs {away_team} will end in a draw!")
        
        st.markdown("</div></div>", unsafe_allow_html=True)
else:
    st.write("No upcoming matches found.")
