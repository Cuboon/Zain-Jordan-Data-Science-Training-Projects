# 📡 Telecom Data Science & ML Pipeline

> **Context:** Built during my internship at **Zain Jordan**. The pipeline reflects real industry workflows; datasets used here are public/synthetic to preserve confidentiality.

This repository walks through a complete, production-style machine learning pipeline applied to telecom data — from raw data wrangling to hyperparameter-tuned classification models. Each notebook is a standalone stage in the workflow, designed to be run sequentially.

---

## 📁 Project Structure

```
notebooks/
├── 01_data_cleaning_and_preprocessing.ipynb   # San Francisco Salaries dataset
├── 02_exploratory_data_analysis.ipynb         # Mall visitor foot traffic (Zain-style telco data)
├── 03_model_training_and_evaluation.ipynb     # Flask API serving salary analysis results
└── 04_model_training_and_hyperparameter_tuning.ipynb  # Churn prediction ML pipeline
```

---

## 🧠 Workflow

### 1️⃣ Data Cleaning & Preprocessing
**`01_data_cleaning_and_preprocessing.ipynb`** · Dataset: `Salaries.csv`

Cleans and explores a public municipal salary dataset, answering business-style questions that mirror payroll analytics done in telecoms.

**What's done:**
- Filled missing `BasePay` and `Benefits` with 0; dropped all-zero rows and negative-pay records
- Year-over-year total salary trend (2011–2014) with growth rate calculation
- Identified the highest mean-paying job title across the full period
- Computed total overtime pay for 2014
- Ranked top 10 highest-earning employees by cumulative `TotalPay`
- Found the 5 least-staffed job titles in 2014 and their associated base pay
- Calculated overtime pay as a percentage of total compensation across all years

**Libraries:** `pandas`, `numpy`, `matplotlib`

---

### 2️⃣ Exploratory Data Analysis
**`02_exploratory_data_analysis.ipynb`** · Dataset: `MALL_VISITORS_UPDATED.csv`

Processes a large-scale telecom subscriber foot-traffic dataset — the kind of location intelligence data Zain would use to understand customer mobility and mall visit behavior.

**What's done:**
- Parsed and typed `VISIT_DATE` as datetime for time-series readiness
- Grouped records by subscriber profile dimensions: `SUBSCRIBER_ID`, `MALL_NAME`, `HOME_GOVERNORATE`, `WORK_GOVERNORATE`, `NATIONALITY`, `ARPU_BRACKET`, `DEVICE_OS`, `AGE_BRACKET`, `GENDER`, `STUDENT`, `IS_EMPLOYEE`, `CREDIT_CATEGORY`
- Aggregated min/max visit timestamps per visit session to derive dwell-time features
- Downsampled to 1,000,000 records for manageable analysis
- Exported cleaned dataset to `Mall_Records.xls` for downstream use

**Libraries:** `pandas`, `numpy`

---

### 3️⃣ Flask API — Results Service
**`03_model_training_and_evaluation.ipynb`** · Dataset: `Salaries.csv`

Wraps the salary analysis functions from Notebook 1 into a **Flask web API**, exposing results as an endpoint — simulating how analytics outputs would be served to a business dashboard or internal tool.

**API functions exposed:**
| Function | Description |
|---|---|
| `calculate_salary_increase()` | Difference in total pay between 2011 and 2014 |
| `highest_mean_salary_job()` | Job title with the highest average pay |
| `calculate_overtime_savings_2014()` | Total overtime pay spend in 2014 |
| `top_earning_employees()` | Top 10 employees by cumulative pay |
| `least_common_jobs_cost_2014()` | Pay cost of the 5 rarest job titles |
| `overtime_pay_percentage()` | Overtime as % of total compensation |

**Libraries:** `flask`, `pandas`

---

### 4️⃣ Churn Prediction — Model Training & Hyperparameter Tuning
**`04_model_training_and_hyperparameter_tuning.ipynb`** · Dataset: `churn_data.csv`

The core ML notebook. Trains and tunes six classification models on a telecom churn dataset, comparing them rigorously using cross-validated metrics and ROC-AUC.

**Pipeline:**
- Dropped nulls, label-encoded all categorical features, mapped `Churn` → binary (Yes/No → 1/0)
- Standardized features with `StandardScaler`
- Applied `StratifiedKFold` (5 splits) to preserve class balance across folds

**Models trained & tuned with `RandomizedSearchCV`:**

| Model | Tuned Hyperparameters |
|---|---|
| Logistic Regression | `C`, `solver` |
| Linear SVM | `C` |
| SVM (RBF Kernel) | `C`, `gamma` |
| K-Nearest Neighbors | `n_neighbors`, `weights`, `metric` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` |

**Evaluation metrics (all cross-validated):** Accuracy · Precision · Recall · F1-Score · ROC-AUC

**Visualizations:** Churn distribution, correlation heatmap, ROC curve (Random Forest), model comparison bar chart by ROC-AUC

**Libraries:** `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `scipy`

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.12 |
| **Data Wrangling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (LR, SVM, KNN, RF, GB) |
| **API** | Flask |
| **Environment** | Jupyter Notebook (Anaconda) |

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn flask scipy

# Launch notebooks
jupyter notebook
```

Run notebooks in order: `01` → `02` → `03` → `04`

> **Note:** Place the required datasets (`Salaries.csv`, `MALL_VISITORS_UPDATED.csv`, `churn_data.csv`) in the same directory as the notebooks before running.

---

## 📈 Key Outcomes

- Built a reproducible end-to-end ML pipeline mirroring real telecom industry workflows
- Processed and aggregated a 1M-row subscriber location dataset reflecting Zain-style telco data
- Deployed analysis results as a live Flask API endpoint
- Benchmarked 6 ML models on churn prediction with full hyperparameter search and cross-validated evaluation

---

*Developed during an internship at Zain Jordan. Public datasets used in place of proprietary data.*
