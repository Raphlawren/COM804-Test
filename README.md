# Chronic Kidney Disease Prediction with Case-Based Reasoning (CBR)

This repository contains my COM804 project on predicting **Chronic Kidney Disease (CKD)** outcomes using a custom **Case-Based Reasoning (CBR)** system in Python. The dataset has **1,138 patient cases** and **23 features**, with two targets:
- **CKD_Progression** (binary: progressing vs not progressing)
- **CKD_Stage** (multiclass: stages 2–5) :contentReference[oaicite:0]{index=0}

---

## What I built

I implemented a CBR pipeline following the CBR cycle:
**Retrieve → Reuse → Revise → Retain**, encapsulated in a `CBR` class that stores past cases, computes similarity, predicts labels, and learns feature weights. :contentReference[oaicite:1]{index=1}

### Similarity + retrieval (Retrieve)
- **Categorical features:** similarity = 1 if values match, else 0  
- **Continuous features:** similarity is based on normalized distance (Min–Max), converted to a 0–1 similarity score :contentReference[oaicite:2]{index=2}
- A query case is compared against the whole case base and the **top-k most similar** cases are retrieved. :contentReference[oaicite:3]{index=3}

### Prediction (Reuse)
I implemented multiple voting options:
- Majority voting
- Similarity-weighted voting
- Distance-weighted voting (squared weights) :contentReference[oaicite:4]{index=4}

### Learning feature weights with gradient descent (Revise/Retain)
A key part of this project is automatic feature-weight learning using **gradient descent** on a validation set:
- predict using current weights  
- compute error against an “ideal” similarity (1 for correct-label neighbors, 0 otherwise)  
- take gradients and update weights to improve accuracy :contentReference[oaicite:5]{index=5}

---

## Data preparation (summary)

### Missing data
Missingness was analyzed with missingness plots, and the pattern suggested **Not Missing at Random (NMAR)** due to incomplete records. :contentReference[oaicite:6]{index=6}

### Handling missing data
- Dropped **88 rows** with missing values in key columns (CKD_Risk, Protein_Creatinine_Ratio, UPCR_Severity), which also reduced missingness in other correlated features. :contentReference[oaicite:7]{index=7}  
- Used **median imputation** for skewed continuous features. :contentReference[oaicite:8]{index=8}  
- Used **class-conditional mode imputation** for selected categorical features (mode computed within each CKD stage). :contentReference[oaicite:9]{index=9}  

### Outliers
Creatinine outliers were removed using the **IQR method** to avoid dominating distance calculations in CBR. :contentReference[oaicite:10]{index=10}

### Train/test split
Data was split into **70% train / 30% test** with **stratified sampling** to preserve class distribution. :contentReference[oaicite:11]{index=11}

---

## Results

### 1) CKD_Progression (Binary)
- Trained for **300 epochs** to learn feature weights. :contentReference[oaicite:12]{index=12}  
- **Test Accuracy:** **84%**  
- **ROC-AUC:** **0.87** :contentReference[oaicite:13]{index=13}  
- Note: the dataset is imbalanced (around **80.3%** “not progressing” vs **19.7%** “progressing”). :contentReference[oaicite:14]{index=14}

### 2) CKD_Stage (Multiclass)
- Over **300 epochs**, training accuracy improved from **57% → 70%** (loss decreased). :contentReference[oaicite:15]{index=15}  
- Top learned feature weights included:
  - Sex (4.34)
  - Occult_Blood_in_Urine (3.00)
  - BMI (1.44)
  - Creatinine (1.43)
  - Hemoglobin (1.22) :contentReference[oaicite:16]{index=16}  

---

## How to run (typical workflow)

1. Clone the repo
2. Create a virtual environment and install requirements
3. Run the notebook/script in the project folder and point it to the dataset path used in the code

(Exact commands depend on how you structured the repo files.)

---

## Notes
This was implemented as a “from scratch” CBR system (not using a ready-made CBR library), including similarity design, multiple voting rules, and gradient-descent weight learning. :contentReference[oaicite:17]{index=17}
