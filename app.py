import streamlit as st
import pickle
import pandas as pd

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

import joblib  # ✅ ADD THIS
pipe = joblib.load('pipe3.pkl')  # ✅ FIX THIS LINE


st.title("🏏 IPL WIN PREDICTION")

# Team and city selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Select Batting Team", sorted(teams))
with col2:
    bowling_team = st.selectbox("Select Bowling Team", sorted(teams))

selected_city = st.selectbox("Select Host City", sorted(cities))

target = st.number_input("Target Score", min_value=1)

# Match progress
col3, col4, col5 = st.columns(3)
with col3:
    current_score = st.number_input("Current Score", min_value=0)
with col4:
    overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10)

# Prediction trigger
if st.button("Predict Probability"):
    try:
        runs_left = target - current_score
        balls_left = 120 - int(overs_completed * 6)
        wickets_remaining = 10 - wickets_lost
        crr = current_score / overs_completed if overs_completed > 0 else 0
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        st.write("📊 Match Situation", input_df)

        prediction = pipe.predict_proba(input_df)
        win_prob = prediction[0][1]
        loss_prob = prediction[0][0]

        st.subheader("🔮 Win Prediction")
        st.success(f"{batting_team} Win Probability: {round(win_prob * 100)}%")
        st.info(f"{bowling_team} Win Probability: {round(loss_prob * 100)}%")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
