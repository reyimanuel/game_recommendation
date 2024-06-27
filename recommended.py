import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Ganti dengan jalur file dataset Anda
df = pd.read_csv('data_game.csv')

# Mengisi nilai kosong dengan 0 atau nilai rata-rata
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
    'Genre': ''
}, inplace=True)

# Konversi kolom 'User_Score' dari string ke float
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce').fillna(0)
df['User_Score'] = df['User_Score'] * 10  # Karena User_Score asli pada skala 0-10, bukan 0-1

# Mengisi nilai kosong pada 'Critic_Score' dan 'User_Score' dengan nilai rata-rata
df['Critic_Score'] = df['Critic_Score'].replace(0, df['Critic_Score'].mean())
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
    if isinstance(game_genre, str) and isinstance(input_genre, str):
        return 1 if game_genre.lower() == input_genre.lower() else 0
    return 0

def get_recommendation(genre_match_val):
    recommendation_sim.input['genre'] = genre_match_val
    
    try:
        recommendation_sim.compute()
        return recommendation_sim.output['recommendation']
    except:
        return 0  # Nilai default jika ada kesalahan

def recommend_games_by_genre(input_genre, threshold=5):
    df['Genre_Match'] = df['Genre'].apply(lambda x: get_genre_match(x, input_genre))
    
    df['Recommendation'] = df.apply(lambda row: get_recommendation(row['Genre_Match']), axis=1)
    
    recommendations = df[df['Recommendation'] >= threshold]
    return recommendations[['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'Global_Sales', 'Recommendation']]

def get_user_input():
    input_genre = input("Masukkan genre yang diinginkan: ")
    return input_genre

def main():
    input_genre = get_user_input()
    recommended_games = recommend_games_by_genre(input_genre)
    print("Game yang direkomendasikan berdasarkan genre yang diinginkan:")
    print(recommended_games)

if __name__ == "__main__":
    main()
