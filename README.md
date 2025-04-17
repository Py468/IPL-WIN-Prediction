
# 🏏 IPL Win Prediction Model

A machine learning project that predicts the **probability of a team winning** an IPL match in real-time based on live second-innings data like runs left, balls remaining, and wickets in hand.

---

## 📂 Project Overview

This model is trained on historical IPL match data (`matches.csv` and `deliveries.csv`) to predict the outcome of a chase during the second innings using features derived from match progress. It’s ideal for integrating with live match dashboards or building predictive cricket apps.

---

## 🧠 Technologies & Tools

- Python 🐍
- Pandas
- Scikit-learn
- Logistic Regression
- OneHotEncoder
- Pipeline
- Pickle

---

## 📊 Features Engineered

- `batting_team`
- `bowling_team`
- `city`
- `runs_left`
- `balls_left`
- `wickets`
- `total_target_runs`
- `current run rate (CRR)`
- `required run rate (RRR)`
- `result` (label: 1 if batting team wins, else 0)

---

## 🛠️ Steps Involved

1. **Data Preprocessing**  
   - Cleaned inconsistent team names and cities  
   - Removed matches with DLS applied

2. **Feature Engineering**  
   - Derived `CRR`, `RRR`, `runs_left`, `balls_left`, `wickets`

3. **Model Building**  
   - Used `Logistic Regression` within a `Pipeline`  
   - Encoded categorical variables with `OneHotEncoder`

4. **Model Evaluation**  
   - Accuracy evaluated using `sklearn.metrics`

5. **Model Export**  
   - Saved as `pipe3.pkl` using `pickle` for deployment

---

## 📈 Sample Prediction

```python
pipe.predict_proba(X_test)[5]
# Output: [0.72, 0.28] —> 72% chance of losing, 28% chance of winning
