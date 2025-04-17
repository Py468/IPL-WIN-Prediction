import streamlit as st
import pickle
import pandas as pd

# Load model
pipe = pickle.load(open('pipe3.pkl', 'rb'))

# Teams and cities (must match training set)
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bengaluru',
    'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Dharamsala', 'Pune', 'Raipur',
    'Ranchi', 'Abu Dhabi', 'Sharjah', 'Cuttack', 'Visakhapatnam',
    'Mohali', 'Bengaluru'
]

# Streamlit UI
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏")
st.title("🏏 IPL Win Prediction")

# Team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Select the Batting Team", sorted(teams))
with col2:
    possible_bowling_teams = [team for team in teams if team != batting_team]
    bowling_team = st.selectbox("Select the Bowling Team", sorted(possible_bowling_teams))

# City selection
selected_city = st.selectbox("Select Host City", sorted(cities))

# Match Inputs
target = st.number_input("Target Score", min_value=1)
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input("Current Score", min_value=0, max_value=target)
with col4:
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_out = st.slider("Wickets Fallen", min_value=0, max_value=10)

# Predict Button
if st.button("Predict Probability"):
    if overs == 0:
        st.warning("Overs can't be zero. Please enter a valid number of overs.")
    else:
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets = 10 - wickets_out
        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Input to model
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # One-Hot Encode categorical variables if necessary
        input_df_encoded = pd.get_dummies(input_df, columns=['batting_team', 'bowling_team', 'city'])

        try:
            result = pipe.predict_proba(input_df_encoded)
            win = result[0][1]
            loss = result[0][0]

            st.subheader("📊 Win Probability")
            st.success(f"{batting_team} 🟢: {win * 100:.2f}%")
            st.error(f"{bowling_team} 🔴: {loss * 100:.2f}%")
        except Exception as e:
            st.error("Something went wrong during prediction.")
            st.exception(e)
