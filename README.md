# Game Recommendation System

A Python-based game recommendation system that uses **Fuzzy Logic** to suggest video games to users based on their preferences. The system is available both as a command-line application and a graphical user interface (GUI).

---

## Project Overview

### Purpose

The Game Recommendation System helps users discover video games that match their preferences by applying a fuzzy inference engine. Rather than returning simple exact-match results, the system computes a continuous **recommendation score** for each game using fuzzy membership functions, allowing for nuanced, ranked suggestions.

### Problem It Solves

With thousands of video game titles available across multiple platforms, finding games that match a player's taste can be overwhelming. This system filters and scores games from a large dataset based on criteria such as genre, platform, minimum user score, and ESRB rating, presenting only the most relevant recommendations.

### Target Audience

- Casual and hardcore gamers looking for new titles to play
- Developers learning how to apply fuzzy logic to recommendation problems
- Data science and AI students studying practical implementations of fuzzy inference systems

---

## Features

- **Fuzzy Logic Engine** — Uses a Mamdani-style fuzzy inference system (via `scikit-fuzzy`) to compute a recommendation score for each game
- **Genre-Based Filtering** — Matches games to the user's preferred genre using fuzzy membership functions
- **Multi-Criteria Filtering (GUI)** — Supports filtering by genre, platform, minimum user score, and ESRB rating simultaneously
- **Large Dataset** — Operates on a dataset of over 16,700 game entries with sales, score, publisher, and rating data
- **CLI Mode** — Lightweight command-line interface for quick genre-based queries
- **GUI Mode** — Desktop graphical interface built with PyQt5 for interactive exploration

---

## Technology Stack

| Category        | Technology                        |
|-----------------|-----------------------------------|
| Language        | Python 3.x                        |
| Fuzzy Logic     | `scikit-fuzzy` (`skfuzzy`)        |
| Data Processing | `pandas`, `numpy`                 |
| GUI Framework   | `PyQt5`                           |
| Dataset         | CSV (Video Game Sales dataset)    |

---

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│                     User Input                        │
│  (Genre / Platform / Min User Score / ESRB Rating)   │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│              Fuzzy Inference Engine                   │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Antecedent: genre (match / not_match)          │ │
│  │  Consequent: recommendation (low / medium /high)│ │
│  │  Rules:                                         │ │
│  │    IF genre IS match     → recommendation HIGH  │ │
│  │    IF genre IS not_match → recommendation LOW   │ │
│  └─────────────────────────────────────────────────┘ │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│              Dataset Filter (pandas)                  │
│  data_game.csv  →  Apply recommendation threshold    │
│                 →  Filter by platform / score / ESRB │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│                  Output Layer                         │
│  CLI: printed DataFrame  │  GUI: QTableWidget        │
└──────────────────────────────────────────────────────┘
```

### How Components Interact

1. **Data Layer** — `data_game.csv` is loaded into a `pandas` DataFrame at startup. Missing values are filled with zeros or empty strings, and `User_Score` is normalized to a 0–100 scale.
2. **Fuzzy Engine** — `scikit-fuzzy` control system defines genre-match membership functions and inference rules. For each game row, `get_recommendation()` feeds the genre-match value into the simulation and retrieves a recommendation score (0–10).
3. **Filter Layer** — The scored DataFrame is filtered by the recommendation threshold and any additional criteria (platform, user score, ESRB rating).
4. **Presentation Layer** — Results are either printed to the terminal (`recommended.py`) or displayed in a `QTableWidget` (`recommended_gui.py`).

---

## Project Structure

```
game_recommendation/
├── data_game.csv          # Video game dataset (~16,700 rows)
├── recommended.py         # Command-line recommendation script
└── recommended_gui.py     # PyQt5 GUI application
```

| File                  | Description                                                                                      |
|-----------------------|--------------------------------------------------------------------------------------------------|
| `data_game.csv`       | Source dataset containing game name, platform, release year, genre, publisher, sales figures, critic/user scores, developer, and ESRB rating |
| `recommended.py`      | CLI entry point. Accepts genre input from the terminal and prints matching game recommendations   |
| `recommended_gui.py`  | GUI entry point. Launches a desktop window where users can filter by genre, platform, user score, and ESRB rating and view results in a table |

---

## Installation

### Requirements

- Python 3.7 or higher
- pip

### Environment Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/reyimanuel/game_recommendation.git
   cd game_recommendation
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install pandas numpy scikit-fuzzy PyQt5
   ```

### Running the Project Locally

**Command-line version:**

```bash
python recommended.py
```

**GUI version:**

```bash
python recommended_gui.py
```

---

## Configuration

The applications are configured directly in the source files. The table below lists the key parameters you can adjust:

| Parameter         | Location               | Default | Description                                              |
|-------------------|------------------------|---------|----------------------------------------------------------|
| `threshold`       | `recommended.py` line 67 / `recommended_gui.py` line 64 | `5`  | Minimum fuzzy recommendation score (0–10) for a game to appear in results |
| Dataset path      | Both files, line 7/8   | `data_game.csv` | Path to the CSV dataset file                   |
| Genre universe    | Both files             | `[0, 2)`      | Fuzzy universe for genre membership (binary: 0 or 1) |
| Recommendation universe | Both files       | `[0, 11)`     | Fuzzy output universe for recommendation score (0–10) |

No external environment variables or `.env` files are required.

---

## Dataset

The dataset (`data_game.csv`) is a video game sales dataset with the following columns:

| Column            | Type    | Description                                    |
|-------------------|---------|------------------------------------------------|
| `Name`            | string  | Game title                                     |
| `Platform`        | string  | Gaming platform (e.g., Wii, PS3, X360)         |
| `Year_of_Release` | integer | Year the game was released                     |
| `Genre`           | string  | Game genre (e.g., Sports, Action, Racing)      |
| `Publisher`       | string  | Publisher name                                 |
| `NA_Sales`        | float   | North America sales (millions)                 |
| `EU_Sales`        | float   | Europe sales (millions)                        |
| `JP_Sales`        | float   | Japan sales (millions)                         |
| `Other_Sales`     | float   | Other regions sales (millions)                 |
| `Global_Sales`    | float   | Total worldwide sales (millions)               |
| `Critic_Score`    | float   | Metacritic critic score (0–100)                |
| `Critic_Count`    | integer | Number of critic reviews                       |
| `User_Score`      | float   | Metacritic user score (normalized to 0–100)    |
| `User_Count`      | integer | Number of user reviews                         |
| `Developer`       | string  | Developer name                                 |
| `Rating`          | string  | ESRB rating (e.g., E, T, M)                    |

---

## Usage

### CLI Mode (`recommended.py`)

```
$ python recommended.py
Masukkan genre yang diinginkan: Sports
Game yang direkomendasikan berdasarkan genre yang diinginkan:
              Name Platform  Year_of_Release   Genre   Publisher  Global_Sales  Recommendation
0       Wii Sports      Wii           2006.0  Sports    Nintendo         82.53            10.0
3  Wii Sports Resort    Wii           2009.0  Sports    Nintendo         32.77            10.0
...
```

Enter a genre (e.g., `Sports`, `Action`, `Racing`, `RPG`) when prompted. The system will print all games with a fuzzy recommendation score of 5 or higher.

### GUI Mode (`recommended_gui.py`)

1. Launch the application: `python recommended_gui.py`
2. Select your preferred **Genre** from the dropdown
3. Select the target **Platform** from the dropdown
4. Enter a **Minimum User Score** (0–10)
5. Select the **ESRB Rating** from the dropdown
6. Click **Recommend** to populate the results table

---

## Development Guide

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your descriptive message"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request against the `main` branch

### Coding Conventions

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use descriptive variable and function names
- Keep fuzzy rule definitions separate from data loading logic
- Avoid mutating the global DataFrame inside filter functions; prefer working on a copy

### Running Tests

There is no automated test suite yet. To manually verify the system:

```bash
# Test CLI output for a known genre
python -c "
from recommended import recommend_games_by_genre
result = recommend_games_by_genre('Sports')
print(result.head())
assert len(result) > 0, 'Should return results for Sports genre'
print('Test passed.')
"
```

---

## Future Improvements

- **Expanded Fuzzy Inputs** — Incorporate `Critic_Score` and `Global_Sales` as additional fuzzy antecedents to produce richer recommendation scores
- **Multi-Genre Support** — Allow users to specify more than one genre preference
- **Weighted Rules** — Add rule weights to the fuzzy system so that score and sales can influence the recommendation alongside genre matching
- **Web Interface** — Replace the PyQt5 GUI with a Flask or FastAPI web application for broader accessibility
- **Automated Tests** — Add a `pytest` test suite covering the fuzzy engine, data loading, and filtering logic
- **Dependency File** — Add a `requirements.txt` or `pyproject.toml` for reproducible environment setup

---

## License

This project does not currently specify a license. Please contact the repository owner for usage permissions.
