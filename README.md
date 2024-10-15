# Premier-League-predictor
Premier League Match Predictor
This project is an English Premier League (EPL) match outcome predictor that leverages machine learning to forecast match results. It uses historical match data sourced from Kaggle and various team statistics to predict whether the home team will win, lose, or draw.

Features
Data Source: The model fetches historical EPL match data from the Kaggle API.
Exploratory Data Analysis (EDA): The program computes key team statistics (e.g., goals scored, shots, fouls, yellow/red cards) to understand each team’s form over their last 5 home and away matches.
Prediction Algorithm: A RandomForestClassifier from scikit-learn is used to predict match outcomes based on the historical performance of the teams.
Flask Web App: The predictor is built as a web application where users can input two teams, and the model will predict the result of the matchup.
Interactive Interface: Users can interact with the app via a simple and intuitive web interface to select teams and view predictions.