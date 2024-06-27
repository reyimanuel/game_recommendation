import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from PyQt5 import QtWidgets, QtGui

# Baca dataset
df = pd.read_csv('data_game.csv')

# Mengisi nilai kosong dengan 0 atau nilai yang sesuai
df.fillna({
    'Critic_Score': 0,
    'User_Score': 0,
    'NA_Sales': 0,
    'EU_Sales': 0,
    'JP_Sales': 0,
    'Other_Sales': 0,
    'Global_Sales': 0,
    'Publisher': '',
    'Developer': '',
    'Rating': '',
    'Genre': '',
    'Platform': '',
}, inplace=True)

# Konversi kolom 'User_Score' dari string ke float dan skala 0-10
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce').fillna(0) * 10

# Mengisi nilai kosong pada 'User_Score' dengan nilai rata-rata
df['User_Score'] = df['User_Score'].replace(0, df['User_Score'].mean())

# Definisikan variabel input dan output
genre = ctrl.Antecedent(np.arange(0, 2, 1), 'genre')
recommendation = ctrl.Consequent(np.arange(0, 11, 1), 'recommendation')

# Definisikan fungsi keanggotaan fuzzy untuk Genre (binary untuk kesederhanaan)
genre['match'] = fuzz.trimf(genre.universe, [1, 1, 1])
genre['not_match'] = fuzz.trimf(genre.universe, [0, 0, 0])

# Definisikan fungsi keanggotaan fuzzy untuk Recommendation
recommendation['low'] = fuzz.trimf(recommendation.universe, [0, 0, 5])
recommendation['medium'] = fuzz.trimf(recommendation.universe, [0, 5, 10])
recommendation['high'] = fuzz.trimf(recommendation.universe, [5, 10, 10])

# Aturan fuzzy
rule1 = ctrl.Rule(genre['match'], recommendation['high'])
rule2 = ctrl.Rule(genre['not_match'], recommendation['low'])

# Membuat sistem kontrol
recommendation_ctrl = ctrl.ControlSystem([rule1, rule2])
recommendation_sim = ctrl.ControlSystemSimulation(recommendation_ctrl)

def get_genre_match(game_genre, input_genre):
    return 1 if isinstance(game_genre, str) and isinstance(input_genre, str) and game_genre.lower() == input_genre.lower() else 0

def get_recommendation(genre_match_val):
    recommendation_sim.input['genre'] = genre_match_val
    try:
        recommendation_sim.compute()
        return recommendation_sim.output['recommendation']
    except:
        return 0  # Nilai default jika ada kesalahan

def recommend_games(input_genre, input_platform, min_user_score, esrb_rating, threshold=5):
    df['Genre_Match'] = df['Genre'].apply(lambda x: get_genre_match(x, input_genre))
    df['Recommendation'] = df.apply(lambda row: get_recommendation(row['Genre_Match']), axis=1)
    
    recommendations = df[
        (df['Recommendation'] >= threshold) &
        (df['Platform'].str.contains(input_platform, case=False, na=False)) &
        (df['User_Score'] >= min_user_score) &
        (df['Rating'].str.contains(esrb_rating, case=False, na=False))
    ]
    return recommendations[['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'Recommendation']]

class GameRecommenderApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Game Recommender System')

        # Genre
        genre_label = QtWidgets.QLabel('Genre:', self)
        genre_label.move(20, 20)
        self.genre_combo = QtWidgets.QComboBox(self)
        self.genre_combo.addItems(sorted(df['Genre'].unique()))
        self.genre_combo.move(150, 20)

        # Platform
        platform_label = QtWidgets.QLabel('Platform:', self)
        platform_label.move(20, 60)
        self.platform_combo = QtWidgets.QComboBox(self)
        self.platform_combo.addItems(sorted(df['Platform'].unique()))
        self.platform_combo.move(150, 60)

        # Min User Score
        min_user_score_label = QtWidgets.QLabel('Min User Score (0-10):', self)
        min_user_score_label.move(20, 100)
        self.min_user_score_entry = QtWidgets.QLineEdit(self)
        self.min_user_score_entry.setValidator(QtGui.QDoubleValidator(0.0, 10.0, 1))
        self.min_user_score_entry.move(150, 100)

        # ESRB Rating
        esrb_rating_label = QtWidgets.QLabel('ESRB Rating:', self)
        esrb_rating_label.move(20, 140)
        self.esrb_rating_combo = QtWidgets.QComboBox(self)
        self.esrb_rating_combo.addItems(sorted(df['Rating'].unique()))
        self.esrb_rating_combo.move(150, 140)

        # Recommend Button
        recommend_button = QtWidgets.QPushButton('Recommend', self)
        recommend_button.move(150, 180)
        recommend_button.clicked.connect(self.recommend)

        # Results
        self.results = QtWidgets.QTableWidget(self)
        self.results.setRowCount(0)
        self.results.setColumnCount(6)
        self.results.setHorizontalHeaderLabels(['Name', 'Platform', 'Year of Release', 'Genre', 'Publisher', 'Recommendation'])
        self.results.move(20, 220)
        self.results.resize(550, 300)

        self.setGeometry(300, 300, 600, 600)
        self.show()

    def recommend(self):
        input_genre = self.genre_combo.currentText()
        input_platform = self.platform_combo.currentText()
        min_user_score = float(self.min_user_score_entry.text())
        esrb_rating = self.esrb_rating_combo.currentText()

        recommended_games = recommend_games(input_genre, input_platform, min_user_score, esrb_rating)

        # Clear previous results
        self.results.setRowCount(0)

        # Insert new results
        for index, row in recommended_games.iterrows():
            row_position = self.results.rowCount()
            self.results.insertRow(row_position)
            self.results.setItem(row_position, 0, QtWidgets.QTableWidgetItem(row['Name']))
            self.results.setItem(row_position, 1, QtWidgets.QTableWidgetItem(row['Platform']))
            self.results.setItem(row_position, 2, QtWidgets.QTableWidgetItem(str(row['Year_of_Release'])))
            self.results.setItem(row_position, 3, QtWidgets.QTableWidgetItem(row['Genre']))
            self.results.setItem(row_position, 4, QtWidgets.QTableWidgetItem(row['Publisher']))
            self.results.setItem(row_position, 5, QtWidgets.QTableWidgetItem(str(row['Recommendation'])))

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    ex = GameRecommenderApp()
    app.exec_()