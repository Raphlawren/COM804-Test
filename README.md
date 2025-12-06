# COM804-Test
Building a CBR model for Chronic Kidney Disease

## EXPLORATORY DATA ANALYSIS (EDA) & PREPROCESSING
#### Dataset
This study uses a Chronic Kidney Disease (CKD) dataset to preridct disease progression and
stage. The dataset contains two target variables: CKD_Progression, a binary indicator of
whether a disease is advancing and CKD_Stage a multiclass variable representing the current
disease stage (2 - 5). The primary objective is to apply Case-Based Reasoninng (CBR), a lazy
learner AI technique that predicts outcomes for new cases based on its similarity to previously
solved past cases.
The dataset comprises 1138 patient cases and 23 features.

1. MISSING DATA
I quantified the missingness per feature and visualized patterns using missingno (matrix +
heatmap).
I calculated the missing values and the percentage of missing values in the dataset.

- BMI has the highest missing value of 137 missing rows (12.04% of the entire dataset)
- Protein_Creatinine_Ratio, CKD_Risk, UPCR_Severity each have 88 missing rows
(7.7%)
- Systolic_Pressure, Dipstick_Proteinuria, Proteinuria each have 16 missing rows
(1.41%).

To understand the pattern of missingness, missingno matrix and heatmap visualizations were
generated. In the matrix, it indicates that where there are values of 1, there are missing values
on the same row, while the heatmap indicated a strong correlation in missingness among
Systolic_Pressure, Hemoglobin, Albumin, CKD_Risk, Dipstick_Poteinuria, Proteinuria,
Occult_Blood_in_Urine, Protein_Creatinine_Ratio, CKD_Risk, and UPCR_Severity. I observed
many features share missing rows for the same patient records. This pattern suggests the data
is Not Missing at Random (NMAR). This is likely due to incomplete medical records; imputing
these would likely inject bias.
