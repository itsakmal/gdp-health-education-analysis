# The Impact of Health and Education on Economic Prosperity: A Causal Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-health-education-analysis-gdkxzzyemojingttl5hkry.streamlit.app/)

This project investigates the complex relationship between a nation's health and education systems and its economic prosperity, measured by GDP per capita. It moves beyond simple correlation by employing a two-pronged approach:

1.  An **exploratory machine learning model** to identify the most predictive indicators within a broad set of health and education variables.
2.  A **rigorous causal inference model** (Two-Way Fixed Effects) to estimate the specific economic impact of key indicators while controlling for confounding factors.

The project culminates in an interactive "what-if" web application that allows users to simulate the economic impact of policy changes based on the final model's findings.

---

## 🚀 Live Demo

You can interact with the final "what-if" simulation model live on Streamlit Community Cloud:

**[https://gdp-health-education-analysis-gdkxzzyemojingttl5hkry.streamlit.app/](https://gdp-health-education-analysis-gdkxzzyemojingttl5hkry.streamlit.app/)**

---

## 📖 Project Overview

The question of what drives economic growth is central to policy and development. While it's widely believed that health and education are key, their specific, measurable impact can be obscured by a web of confounding variables (e.g., urbanization, macroeconomic stability) and short-term economic momentum.

This project aims to answer the question: **"What is the independent, causal impact of specific health and education indicators on a nation's GDP per capita?"**

To achieve this, we first use a machine learning model (LightGBM) as a powerful feature selection tool to discover the most promising indicators from a large dataset. We then use these indicators as "treatments" in a Two-Way Fixed Effects (TWFE) regression model, a standard econometric technique for establishing more causal relationships from panel data. This allows us to isolate the effect of our variables of interest from time-invariant country characteristics and global shocks that affect all countries in a given year.

---

## ✨ Key Findings

Our final, robust causal model reveals two significant drivers of economic prosperity:

1.  **Sustained Health of the Adult Population is Paramount**: The most powerful predictor is **Life Expectancy at Age 60**. A **one-year increase** in this metric is associated with a remarkable **$2,031 increase** in GDP per capita, holding all other factors constant. This highlights the profound economic importance of a healthcare system that supports a long, healthy, and productive life for its experienced citizens.

2.  **Higher Education is a Significant Economic Engine**: **Tertiary Enrollment** also has a strong, positive impact. A **one-percentage-point increase** in the gross tertiary enrollment rate is associated with a **$61 increase** in GDP per capita.

The full regression results, controlling for factors like age dependency, urbanization, and inflation, confirm that these relationships are statistically significant and robust.

---

## 🛠️ Methodology

The project follows a structured, multi-stage pipeline.

### 1. Data Source

The analysis is based on the World Bank's Global Development Statistics dataset.

* **Source**: [World Bank Data360 - Wide Format CSV](https://data360files.worldbank.org/data360-data/data/WB_GS/WB_GS_WIDEF.csv)
* **Scope**: The dataset contains hundreds of indicators for over 200 countries, spanning from 1980 to the present.

### 2. Preprocessing (`src/preprocess.py`)

A curated list of relevant health, education, and economic control indicators is defined in `configs/selected_indicators.csv`. The preprocessing script performs the following steps:
* Reads the raw, wide-format CSV.
* Filters the data to keep only the selected indicators.
* Melts the data into a long format (one row per country-year-indicator).
* Pivots the data into a panel format (one row per country-year).
* Engineers features by creating time-lagged (`_lagX`) and change-over-time (`_chgX`) versions of each indicator.
* Saves the final, clean dataset as `data/processed/panel_clean.parquet`.

### 3. Exploratory Analysis (`src/train.py`)

To identify the most promising indicators, a machine learning model (LightGBM) is trained to predict GDP per capita using only health and education variables. This acts as a sophisticated feature selection step, revealing which indicators have the strongest predictive power once obvious economic confounders are removed.

### 4. Causal Inference (`src/causal.py`)

This is the core of the analysis. A Two-Way Fixed Effects (TWFE) regression model is implemented using `statsmodels`.
* **Model**: `GDPpc ~ β1*Health + β2*Education + Controls + CountryEffects + YearEffects`
* **Purpose**: This isolates the effects of our `treatments` (health and education) from:
    * **Country Fixed Effects**: Stable, unobserved characteristics of a country (e.g., culture, geography).
    * **Year Fixed Effects**: Global shocks that affect all countries in a given year (e.g., a global recession).
* **Output**: The script produces robust coefficient estimates (`β`) that form the basis of the Streamlit "what-if" tool.

---

## ⚙️ How to Run Locally

To replicate the analysis and run the Streamlit app on your own machine, follow these steps.

### Prerequisites
* Python 3.9+
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/itsakmal/gdp-health-education-analysis.git](https://github.com/itsakmal/gdp-health-education-analysis.git)
cd gdp-health-education-analysis
```

### 2. Set Up a Virtual Environment
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4.  Run the Full Pipeline
```bash
# Step 1: Preprocess the raw data
python src/preprocess.py

# Step 2 (Optional): Run the exploratory ML model
python src/train.py

# Step 3: Run the causal analysis to generate coefficients
python src/models/causal.py

# Step 4: Launch the interactive web app
streamlit run streamlit_app.py
```
The application should now be running and accessible in your web browser.

## File Structure
```bash
gdp-health-education-analysis/
│
├── configs/
│   ├── config.yaml
│   └── selected_indicators.csv
│
├── data/
│   ├── raw/
│   │   └── WB_GS_WIDEF.csv   # raw dataset
│   └── processed/
│       └── panel_clean.parquet
│
├── notebooks/
│   ├── reports/figures
│   └── 01_eda.ipynb
│
├── src/
│   ├── models/
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── causal.py
│   │   └── causal_params_*.csv
│   └── utils/
│       └── visualization/plots.py
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

## 💻 Technologies Used  

**Language:**  
- Python  

**Core Libraries:**  
- **Data Manipulation:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn, LightGBM  
- **Econometrics:** Statsmodels  
- **Web Application:** Streamlit  
- **Configuration:** PyYAML  
- **Data Format:** Parquet (via FastParquet)

## 📜 License
This project is licensed under the [MIT License](./LICENSE.txt).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE.txt)
