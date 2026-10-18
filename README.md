# Formula 1 Podium Prediction Companion — Milestone 1

Predicting whether a Formula 1 driver finishes on the podium (top 3) using strictly pre-race information known after qualifying and before the race starts.

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/)

---

## 📌 Project Overview
* **Unit of Analysis**: One driver entering one race (`raceId`, `driverId`).
* **Target**: Binary classification `podium = (positionOrder <= 3)` (~13.61% cleaned base rate).
* **Prediction Cutoff**: Post-qualifying, pre-race start. Current-race results, points, laps, pit stops, and status are strictly excluded.
* **Leakage Verification**: Feature set frozen (`features_v1.json`) and verified via own-race flip tests, control tests, and teammate consistency checks.
* **Dataset**: Formula 1 World Championship (1950–2024) — Kaggle / Ergast relational dataset (14 relational tables).

---

## 🏎️ Key Data-Engineering Findings

1. **Q1: Circuit Front-Row Podium Conversion (2003–2013 vs. 2014–2024)**
   * Suzuka, Yas Marina, and Monaco exhibit the highest front-row podium conversion post-2014.
   * Spearman rank correlation between eras across the 14 qualifying circuits is $\rho = 0.30$ ($p = 0.29$) by grid position and $\rho = 0.57$ ($p = 0.03$) by qualifying position.
   * Pooled front-row conversion significantly increased in the hybrid era from **67.0% to 76.7%** (grid, Fisher exact $p = 0.014$).

2. **Q2: Home Grand Prix Advantage**
   * After controlling for starting grid position across four distinct analyses (5 grid bins, Mantel-Haenszel stratified on exact grid slot, within-driver pairing, and clustered logistic GLM), there is **no statistically significant home-country podium advantage** (Odds Ratio: 0.96, 95% CI: 0.79–1.16, $p = 0.67$).
   * The lower raw home podium rate (11.9% vs. 13.8%) was driven by home drivers qualifying further back on average (mean grid 12.7 vs. 11.8).

3. **Q3: Constructor Mechanical Reliability (2014–2021 vs. 2022–2024)**
   * Mechanical retirements fell significantly across the hybrid era from **9.24% to 5.69%** across all starts ($p < 0.001$), and from **9.15% to 5.69%** for the 10 continuous constructor lineages ($p < 0.001$).
   * With Benjamini–Hochberg multiple-testing correction, **McLaren** ($13.8\% \to 3.7\%$, $p_{\text{adj}} = 0.008$) and **RB / Toro Rosso** ($14.7\% \to 5.9\%$, $p_{\text{adj}} = 0.037$) demonstrated statistically significant reliability gains.

---

## 📊 Model Evaluation & Results

Data split chronologically into:
* **Train**: 1950–2019 (22,553 rows, 13.47% podium rate)
* **Validation**: 2020–2021 (779 rows, 15.02% podium rate)
* **Test**: 2022–2024 (1,354 rows, 15.07% podium rate)

Due to class imbalance, **PR-AUC (Average Precision)** serves as the primary metric, with **F1 score** as secondary. Thresholds were selected strictly on the validation set.

### Test Set Performance (2022–2024, Unseen)

| Model Family | ROC-AUC | PR-AUC (Primary) | F1 Score (Secondary) | Precision | Recall | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Grid $\le$ 3)** | 0.914 | 0.647 | 0.649 | 0.650 | 0.647 | 0.894 |
| **Logistic Regression** | 0.932 | 0.686 | 0.686 | 0.618 | 0.770 | 0.894 |
| **Random Forest (Chosen)** | **0.936** | **0.698** | **0.665** | **0.679** | **0.652** | **0.901** |
| **HistGradientBoosting** | 0.936 | 0.693 | 0.672 | 0.606 | 0.755 | 0.889 |
| **Shallow Neural Network** | 0.929 | 0.687 | 0.669 | 0.568 | 0.814 | 0.879 |

* **Chosen Model**: Tuned Random Forest (`max_features=0.5`, `min_samples_leaf=20`, decision threshold = `0.787` selected via max validation F1).

---

## 🔍 Model Explainability (XAI)
* **Global Explanations**: Permutation feature importance and SHAP summary beeswarm show that starting position (`grid_clean` and `quali_position`) and constructor form (`constructor_podiums_last5`) dominate predictions.
* **Local Explanations**: SHAP waterfall and LIME explain individual race predictions (e.g. Verstappen 2022 Canadian GP hit vs. Gasly 2023 Dutch GP miss).
* **Non-Causal Notice**: Feature importance describes statistical reliance within the model and does not establish physical or sporting causation.

---

## 📁 Repository Structure

```text
├── README.md                      # Project documentation and summary
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore file
├── features_v1.json               # Frozen 17-feature registry
├── full_metrics_table.csv         # Consolidated model performance across splits
├── notebooks/
│   └── m1-markups.ipynb           # Complete, run-all analysis and modeling notebook
├── figures/                       # Generated analysis and diagnostic plots
│   ├── Figure_1_Top_Sentinels.png
│   ├── Figure_2_Data_Cleaning_Waterfall.png
│   ├── Figure_3_Q1_Circuit_Caterpillar_Wilson_CI.png
│   ├── Figure_4_Q2_Home_Advantage_Analysis.png
│   ├── Figure_5_Feature_Podium_Rate_Distributions.png
│   ├── Figure_6_Model_PR_Curve_and_Confusion_Matrix.png
│   ├── Figure_7_LIME_Local_Explanations.png
│   ├── Figure_7_SHAP_Summary_Beeswarm.png
│   ├── Figure_7_SHAP_Waterfall_Cases.png
│   ├── Figure_Grid_vs_Podium_Rate.png
│   ├── Figure_Q3_Mechanical_Retirements_Caterpillar.png
│   └── Figure_Qualifying_Coverage_Drift.png
└── data/                          # Ergast F1 relational dataset (14 CSVs)
```

---

## 🚀 Reproduction & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Launch Jupyter Notebook and open `notebooks/m1-markups.ipynb`:
```bash
jupyter notebook notebooks/m1-markups.ipynb
```
The notebook executes top-to-bottom without manual interventions. Step 3.7 automatically runs a test suite verifying all reported metrics and figures.

---

## 👥 Team Contributions
* **Member 1**: Data cleaning, sentinel handling, relational schema integrity, Q1 & Q2 statistical analyses.
* **Member 2**: Pre-race feature engineering, chronological data splitting, Q3 reliability analysis, report authoring.
* **Member 3**: Statistical ML and neural network pipelines, threshold tuning, XAI (SHAP & LIME), inference API.
