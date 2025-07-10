# Creditworthiness Prediction Using Random Forest

![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-brightgreen)
![Python](https://img.shields.io/badge/Language-Python%203.8%2B-yellow)

## 📌 Project Overview

This project aims to predict the **creditworthiness** of individuals based on various financial and personal attributes using the **German Credit Dataset**. The primary goal is to build a reliable machine learning pipeline using the **Random Forest Ensemble Method** that can classify whether a person is a **good (1)** or **bad (2)** credit risk.

---

## 📂 Repository Structure
📁 Creditworthiness-RandomForest/  
│  
├── README.md # Project description and instructions  
├── german.data # Raw dataset (categorical + numeric)  
├── german.data-numeric # Cleaned, fully numeric version of the dataset  
├── german.doc # Attribute definitions and metadata  
├── file.ipynb # Full Jupyter Notebook with EDA, modeling, and evaluation  
└── requirements.txt # Dependencies  

---

## 🧠 Problem Statement

Financial institutions need to assess an individual's **credit risk** before approving loans or credit cards. Misjudging creditworthiness can lead to financial losses.

- **Goal**: Predict whether a customer is likely to **default** or **repay** based on 24 financial and demographic attributes.
- **Challenge**: 
  - Mixed data types (categorical + numerical)
  - Imbalanced classes
  - Cost-sensitive nature of misclassification:
    - Mistaking a bad customer as good is **5× costlier** than the opposite.

---

## 📊 Dataset Description

- **`german.data`**: Original dataset with 20 attributes (13 categorical, 7 numerical)
- **`german.data-numeric`**: Transformed dataset with 24 fully numeric attributes + label
- **`german.doc`**: Contains detailed attribute descriptions and cost matrix

> Total records: `1000`  
> Classes:  
> - `1` = Good Credit Risk  
> - `2` = Bad Credit Risk

---

## 🔬 Approach & Solution

### 🔁 ML Pipeline Overview

1. **Data Loading**  
   Load `german.data-numeric` using `pandas`, assign appropriate column names.

2. **Exploratory Data Analysis (EDA)**  
   - Class imbalance visualization  
   - Feature correlation analysis  
   - Statistical summaries

3. **Preprocessing**  
   - Feature scaling using `StandardScaler`  
   - Train-Test Split (80-20)

4. **Model Building**  
   - Model: `RandomForestClassifier`  
   - Fit on training set  
   - Predict on test set

5. **Evaluation**  
   - Confusion Matrix  
   - Classification Report (Precision, Recall, F1-Score)  
   - ROC-AUC Score  
   - Cost-Sensitive Error Calculation using a custom cost matrix

6. **Feature Importance Analysis**  
   - Plot top contributing features

---

## ⚙️ Tech Stack

| Tool            | Description                           |
|-----------------|---------------------------------------|
| Python 3.8+     | Programming Language                  |
| Jupyter Notebook| Interactive data analysis             |
| Pandas          | Data loading and preprocessing        |
| Seaborn/Matplotlib | Data visualization                |
| Scikit-learn    | ML modeling and evaluation            |

---

## 📈 Key Results

| Metric                  | Value      |
|-------------------------|------------|
| Accuracy                | 77%      |
| ROC AUC (Good=1)        | 0.208     |
| Total Cost (Custom Matrix)| Calculated and minimized |
| Top Features            | Duration, Credit Amount, Age, Employment |

---

## 📌 Cost Matrix Consideration

As defined in `german.doc`, the business risk is asymmetric:

| Actual \ Predicted | Good (1) | Bad (2) |
|--------------------|----------|---------|
| Good (1)           | 0        | 1       |
| Bad (2)            | 5        | 0       |

This matrix penalizes **false positives (bad predicted as good)** 5× more.

---

## 🧾 How to Run

### 📦 Install dependencies
```bash
pip install -r requirements.txt
```
## Launch Notebook
jupyter notebook file.ipynb

## Insights

- The Random Forest classifier handled the mixed data and class imbalance relatively well.

- Cost-based evaluation revealed significant business implications in prediction errors.

- EDA uncovered highly influential features like credit amount, duration, and employment status.

## Conclusion
This project successfully built a creditworthiness prediction system using the Random Forest Ensemble Method on the German Credit dataset. With thorough preprocessing, thoughtful evaluation, and cost-sensitive analysis, the model provides a strong baseline for risk assessment in financial systems.

