# Dementia Progression Prediction

##  Overview
This project aims to predict the progression stages of dementia using clinical and MRI-based features from the OASIS longitudinal dataset. Early prediction can assist in better diagnosis and treatment planning.

---

##  Problem Statement
Dementia progression is difficult to track manually. This project uses machine learning to classify patients into different stages based on medical data.

---

##  Model Details
- Algorithm Used: XGBoost Classifier
- Type: Multi-class Classification
- Target Classes:
  - No Dementia
  - Very Mild Dementia
  - Dementia

---

##  Dataset
- Source: OASIS Longitudinal Dataset
- Features: Clinical + MRI-based attributes

---

##  Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn

---

##  How to Run

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
3. Run the modeil
```bash
python model.py