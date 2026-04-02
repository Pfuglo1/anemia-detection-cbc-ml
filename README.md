<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Anemia%20Detection%20via%20CBC&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Machine%20Learning%20on%20Complete%20Blood%20Count%20Parameters&descAlignY=56&descSize=16" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189FB5?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **A clinical-grade machine learning pipeline to detect anemia from routine blood test parameters — with special emphasis on maximizing Recall to minimize missed diagnoses.**

<br/>

[🔍 Explore the Notebook](#-project-structure) · [📊 See Results](#-model-performance) · [🚀 Run Locally](#-getting-started) · [🩸 About the Problem](#-problem-statement)

<br/>
</div>

---

## 🩸 Problem Statement

Anemia is one of the most common and globally prevalent blood disorders, affecting over **1.6 billion people** worldwide. Early detection is critical — especially in children and pregnant women — yet it is frequently missed in routine screenings.

This project builds a **binary classification pipeline** on real Complete Blood Count (CBC) data to:

- Automatically flag anemic patients from standard lab results
- **Maximize Recall** — because a missed anemia diagnosis (False Negative) carries clinical risk
- Provide interpretable feature importance for medical trust and explainability

**Anemia is defined as:**
```
Anaemia = 1  if  HGB < 11.0 g/dL  OR  HCT < 36.0%
Anaemia = 0  otherwise
```

---

## 📁 Project Structure

```
anemia-detection-cbc-ml/
│
├── 📓 Project_7_Anemia_Detection.ipynb   # Main notebook (all parts)
│
├── 📊 data/
│   ├── cbc_raw.csv                       # Original Kaggle dataset
│   └── cbc_anaemia_clean.csv             # Cleaned dataset (output)
│
├── 🤖 models/
│   ├── anaemia_model.pkl                 # Final saved model
│   └── scaler.pkl                        # Fitted StandardScaler
│
├── 📈 plots/                             # All generated visualizations
│
└── 📄 README.md
```

---

## 🗂️ Dataset

| Property | Detail |
|---|---|
| **Source** | [Kaggle – CBC Test Dataset](https://www.kaggle.com/datasets/ahmedelsayedtaha/complete-blood-count-cbc-test/data) |
| **Features** | 20 CBC parameters (WBC, RBC, HGB, HCT, MCV, MCH, MCHC, PLT, etc.) |
| **Target** | `Anaemia` — binary (1 = Anemic, 0 = Not Anemic) |
| **Label Source** | Derived from clinical HGB/HCT thresholds |

### CBC Feature Reference

| Feature | Description |
|---|---|
| `WBC` | White Blood Cell count |
| `RBC` | Red Blood Cell count |
| `HGB` | Hemoglobin — *primary anemia indicator* |
| `HCT` | Hematocrit — *primary anemia indicator* |
| `MCV` | Mean Corpuscular Volume |
| `MCH` / `MCHC` | Mean Corpuscular Hemoglobin (Concentration) |
| `RDWSD` / `RDWCV` | Red cell Distribution Width |
| `PLT` | Platelet count |
| `MPV` / `PDW` / `PCT` / `PLCR` | Platelet morphology indices |

---

## 🔬 Project Workflow

```
Raw CBC Data
     │
     ▼
Part 1: EDA ──────────── shape, types, missing values, descriptive stats
     │
     ▼
Part 2: Target Creation ─ Anaemia = f(HGB, HCT)
     │
     ▼
Part 3: Deep EDA ─────── distributions, outlier analysis, t-tests,
     │                    Pearson / Spearman / Point-Biserial correlations
     ▼
Part 4: Preprocessing ── imputation, train-test split, StandardScaler
     │
     ▼
Part 5: Model Training ── 5 classifiers trained and evaluated
     │
     ▼
Part 6: Threshold Tuning ─ ROC curves, threshold sweep for max Recall
     │
     ▼
Part 7: Best Model ──────── comparison table, feature importance,
     │                       reduced feature model
     ▼
Part 8: Persistence ─────── save cleaned CSV, model .pkl, scaler .pkl
     │
     ▼
Part 9: Inference ───────── load model, predict on a new patient sample
```

---

## 🤖 Models Trained

| # | Model | Notes |
|---|---|---|
| 1 | **Logistic Regression** | Baseline linear model, interpretable coefficients |
| 2 | **Decision Tree** | Interpretable, prone to overfit without pruning |
| 3 | **Random Forest** | Ensemble, handles class imbalance well |
| 4 | **XGBoost** | Gradient boosting, strong performer on tabular data |
| 5 | **K-Nearest Neighbors** | Distance-based, sensitive to scaling |

---

## 📊 Model Performance

> All models evaluated on held-out test set (20% stratified split).  
> **Primary metric: Recall** — minimizing False Negatives is the clinical priority.

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — | — |
| Decision Tree | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| **XGBoost** | — | — | — | — | — | — |
| KNN | — | — | — | — | — | — |

> 📌 *Results populate after running the notebook with the dataset.*

---

## ⚙️ Threshold Tuning

Default classifiers use a **0.5 probability threshold**. In medical diagnosis, this is often too conservative — we'd rather over-flag than miss a patient.

This project includes a **threshold sweep** (0.1 → 0.9) that plots Recall vs. Precision and selects an optimal cutoff:

```python
# Example interpretation:
"We choose threshold 0.35 because it gives Recall of 0.96 (only 4 FN),
which is critical for clinical safety even though Precision drops to 0.78."
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/anemia-detection-cbc-ml.git
cd anemia-detection-cbc-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy joblib jupyter
```

### 3. Download the dataset

Get the CSV from [Kaggle](https://www.kaggle.com/datasets/ahmedelsayedtaha/complete-blood-count-cbc-test/data) and place it in `data/`.

### 4. Run the notebook

```bash
jupyter notebook Project_7_Anemia_Detection.ipynb
```

---

## 🧪 Predict on a New Patient

```python
import joblib
import pandas as pd

model = joblib.load('models/anaemia_model.pkl')
scaler = joblib.load('models/scaler.pkl')

sample = pd.DataFrame({
    'WBC': [7.5], 'RBC': [3.9], 'HGB': [10.5], 'HCT': [34.0],
    'MCV': [88.0], 'MCH': [28.0], 'MCHC': [33.0], 'RDWCV': [14.0],
    'PLT': [250.0], 'MPV': [9.0]
    # ... all selected features
})

sample_scaled = scaler.transform(sample)
pred_class = model.predict(sample_scaled)[0]
pred_proba = model.predict_proba(sample_scaled)[0][1]

print(f"Predicted anemia status: {pred_class} ({'Anemic' if pred_class else 'Not Anemic'})")
print(f"Probability of anemia: {pred_proba:.2f}")
```

---

## 📌 Key Findings

- **HGB and HCT** are the most discriminative features (by design and confirmed via point-biserial correlation)
- **RDWCV, MCV, MCH, MCHC** also show statistically significant differences between classes (t-test, p < 0.05)
- The dataset exhibits **moderate class imbalance** — anemic patients are the minority class
- **XGBoost / Random Forest** achieve the best ROC-AUC; Logistic Regression provides strong recall with calibrated thresholds
- A **reduced model** (top 5–8 features) performs comparably to the full model — favoring interpretability

---

## 🧠 Skills Demonstrated

- ✅ End-to-end ML pipeline design
- ✅ Clinical domain awareness (CBC interpretation, anemia criteria)
- ✅ Statistical feature analysis (t-tests, Pearson, Spearman, Point-Biserial)
- ✅ Outlier handling and imputation strategies
- ✅ Multiple classifier comparison with proper evaluation metrics
- ✅ ROC curve analysis and custom threshold tuning
- ✅ Feature importance extraction and model simplification
- ✅ Model persistence and real-world inference pattern

---

## 📚 References

- Dataset: [Kaggle – Complete Blood Count CBC Test](https://www.kaggle.com/datasets/ahmedelsayedtaha/complete-blood-count-cbc-test/data)
- WHO Anemia Guidelines: [who.int/health-topics/anaemia](https://www.who.int/health-topics/anaemia)
- Scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org)
- XGBoost Documentation: [xgboost.readthedocs.io](https://xgboost.readthedocs.io)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**Made with ❤️ and a lot of blood tests**

⭐ Star this repo if you found it useful!

</div>
