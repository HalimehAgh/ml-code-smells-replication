# Hypothesis Testing and Statistical Analysis
 Each script performs statistical hypothesis testing and generates the corresponding analytical outputs and visualizations.

---

## 1. RQ1 Analysis (`rq1_analysis.py`)

This script compares the **density of ML-specific code smells** against **general Python code smells**.

### Hypothesis (H0.0)

- Evaluates whether a **statistically significant difference** exists between:
  - Specialized **ML smell density**
  - General **functional Python smell density**

### Methodology

- Applies a **Paired Wilcoxon Signed-Rank Test** to compare smell densities.
- Compares:
  - **Top 20 functional Pylint smells**
  - **12 ML-specific smell types** detected by CodeSmile

### Required Files

- `combined_analysis_data.csv`  
  *(ML smell counts and density information)*

- `pylint_results/` directories  
  *(Extracted Python smell data)*

- `repos_small.csv`, `repos_medium.csv`, `repos_large.csv`  
  *(Used for Lines of Code normalization)*

---

## 2. RQ2 Analysis (`rq2_analysis.py`)

A master runner script that executes multiple hypothesis tests related to **code smell distribution across different categories**.

### Structure

- Calls sub-analysis scripts located in the `RQ2_tests/` directory.

### Tests Included

- **H1 – H6**  
  Evaluate various correlations, comparisons, and distributional properties of code smells.

### Organizational Note

- Each hypothesis test (**H1** through **H6**) is stored in its **own folder**.
- Every folder contains:
  - The specific `h*.py` script
  - The dedicated CSV datasets required for that individual test

---

## 3. RQ3 Analysis (`rq3_analysis.py`)

This script analyzes code smells in the context of **specific ML frameworks or methodologies**.

### Runner Behavior

- Executes the following sub-tests located in the `RQ3_tests/` directory:
  - `h1.py`
  - `h2h6.py`

### Outputs

- Generates statistical summaries and analytical results addressing **RQ3-specific hypotheses**.

---

## 4. RQ4 Analysis (`rq4_analysis.py`)

Focuses on the relationship between **project characteristics** and **code smell prevalence**.

### Runner Behavior

- Executes six sequential hypothesis tests:
  - **H7 – H12**

### Structure

- Tests are organized in the `RQ4_tests/` directory.
- Each hypothesis (**H7** through **H12**) has:
  - Its own subdirectory
  - A dedicated `h*.py` script
  - The required CSV datasets for that specific analysis

---
Ensure all result CSVs from the **processing phase** are located in the same directory as the RQ scripts.
