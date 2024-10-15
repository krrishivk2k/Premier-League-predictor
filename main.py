import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# Global variables to hold the trained model and team statistics
model = None
team_stats = None

# Function to download data using Kaggle API
def download_data():
    path = kagglehub.dataset_download("irkaal/english-premier-league-results")
    return os.path.join(path, 'results.csv')

# Function to calculate team statistics
def calculate_team_stats(df):
    stats = {}
    for team in set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()):
        home_games = df[df['HomeTeam'] == team].tail(5)
        away_games = df[df['AwayTeam'] == team].tail(5)
        
        stats[team] = {
            'FTHG_Mean': home_games['FTHG'].mean(),
            'FTAG_Mean': away_games['FTAG'].mean(),
            'HS_Mean': home_games['HS'].mean(),
            'AS_Mean': away_games['AS'].mean(),
            'HST_Mean': home_games['HST'].mean(),
            'AST_Mean': away_games['AST'].mean(),
            'HC_Mean': home_games['HC'].mean(),
            'AC_Mean': away_games['AC'].mean(),
            'HF_Mean': home_games['HF'].mean(),
            'AF_Mean': away_games['AF'].mean(),
            'HY_Mean': home_games['HY'].mean(),
            'AY_Mean': away_games['AY'].mean(),
            'HR_Mean': home_games['HR'].mean(),
            'AR_Mean': away_games['AR'].mean(),
            'Points': df[((df['HomeTeam'] == team) & (df['FTR'] == 'H')) | 
                         ((df['AwayTeam'] == team) & (df['FTR'] == 'A'))].shape[0] * 3 +
                      df[((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
                         (df['FTR'] == 'D')].shape[0]
        }
    return stats

# Load, preprocess, and train the model
def load_and_train_model():
    global model, team_stats
    file_path = download_data()
    
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    
    team_stats = calculate_team_stats(df)
    
    # Prepare features and target
    X = []
    y = []
    
    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        if home_team in team_stats and away_team in team_stats:
            home_stats = team_stats[home_team]
            away_stats = team_stats[away_team]
            features = [
                home_stats['FTHG_Mean'], home_stats['FTAG_Mean'],
                away_stats['FTHG_Mean'], away_stats['FTAG_Mean'],
                home_stats['HS_Mean'], away_stats['AS_Mean'],
                home_stats['HST_Mean'], away_stats['AST_Mean'],
                home_stats['HC_Mean'], away_stats['AC_Mean'],
                home_stats['HF_Mean'], away_stats['AF_Mean'],
                home_stats['HY_Mean'], away_stats['AY_Mean'],
                home_stats['HR_Mean'], away_stats['AR_Mean'],
                home_stats['Points'] - away_stats['Points']
            ]
            X.append(features)
            y.append(row['FTR'])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

# Function to make predictions
def predict_match(home_team, away_team):
    if home_team not in team_stats or away_team not in team_stats:
        return "Unknown team(s)"
    
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    features = [
        home_stats['FTHG_Mean'], home_stats['FTAG_Mean'],
        away_stats['FTHG_Mean'], away_stats['FTAG_Mean'],
        home_stats['HS_Mean'], away_stats['AS_Mean'],
        home_stats['HST_Mean'], away_stats['AST_Mean'],
        home_stats['HC_Mean'], away_stats['AC_Mean'],
        home_stats['HF_Mean'], away_stats['AF_Mean'],
        home_stats['HY_Mean'], away_stats['AY_Mean'],
        home_stats['HR_Mean'], away_stats['AR_Mean'],
        home_stats['Points'] - away_stats['Points']
    ]
    prediction = model.predict([features])[0]
    return prediction

@app.route("/", methods=["GET", "POST"])
def index():
    teams = sorted(team_stats.keys()) if team_stats else []
    if request.method == "POST":
        home_team = request.form.get("home_team")
        away_team = request.form.get("away_team")
        result = predict_match(home_team, away_team)
        return render_template("index.html", teams=teams, result=result, home_team=home_team, away_team=away_team)
    return render_template("index.html", teams=teams, result=None)

if __name__ == "__main__":
    load_and_train_model()  # Load and train the model at startup
    app.run(port=8080, debug=True)