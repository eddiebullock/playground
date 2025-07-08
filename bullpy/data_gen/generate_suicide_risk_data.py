import numpy as np
import pandas as pd

def generate_data(n_patients=1000, high_risk_prevalence=0.1, seed=42):
    np.random.seed(seed)
    # Demographics
    ages = np.random.normal(loc=35, scale=12, size=n_patients).astype(int)
    genders = np.random.choice(['Male', 'Female', 'Other'], size=n_patients, p=[0.48, 0.48, 0.04])
    # Clinical: depression score (0-27), higher means more severe
    depression_scores = np.clip(np.random.normal(loc=8, scale=5, size=n_patients), 0, 27).round(1)
    # Risk label: high risk if depression score is high, plus some randomness
    risk = (depression_scores > 16).astype(int)
    # Add some high risk cases randomly to match prevalence
    n_high_risk = int(n_patients * high_risk_prevalence)
    high_risk_indices = np.random.choice(n_patients, n_high_risk, replace=False)
    risk[:] = 0
    risk[high_risk_indices] = 1

    df = pd.DataFrame({
        'age': ages,
        'gender': genders,
        'depression_score': depression_scores,
        'high_risk': risk
    })
    df.to_csv('synthetic_suicide_risk_data.csv', index=False)
    print(f"Generated synthetic data for {n_patients} patients. Saved to 'synthetic_suicide_risk_data.csv'.")

if __name__ == "__main__":
    generate_data() 