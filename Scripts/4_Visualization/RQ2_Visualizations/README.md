# Visualizations


### 1. `part1_visualizations.py`  
**Hypothesis H1: Project Size**

**Purpose**  
Analyzes how ML-specific code smell density varies with project size.

**Required Input**
- `analysis_data.csv`

**Generated Figures**
- `fig_h1_correlations.png`  
  *Correlation between project size metrics and ML smell density*
- `fig_h1_boxplot.png`  
  *Distribution of ML smell density across size categories*

---

### 2. `part2_visualizations.py`  
**Hypothesis H2: Project Age**

**Purpose**  
Examines the relationship between project age and ML-specific code smells.

**Required Input**
- `age_analysis_data.csv`

**Generated Figures**
- `fig_h2_age_correlation.png`  
  *Correlation between project age and ML smell density*
- `fig_h2_age_boxplot.png`  
  *Comparison of smell density across age groups*

---

### 3. `part3_visualizations.py`  
**Hypothesis H3: Team Size**

**Purpose**  
Evaluates whether the number of contributors impacts ML code smell prevalence.

**Required Input**
- `contributors_analysis_data.csv`

**Generated Figures**
- `fig_h3_team_correlation.png`  
  *Correlation between team size and ML smell density*
- `fig_h3_team_boxplot.png`  
  *Distribution of smell density across team-size categories*

---

### 4. `part4_visualizations.py`  
**Hypothesis H4: Maintenance Activity**

**Purpose**  
Investigates the relationship between repository maintenance activity and ML code smells.

**Required Input**
- `commit_activity_data.csv`

**Generated Figures**
- `fig_h4_activity_correlation.png`  
  *Correlation between commit activity and ML smell density*
- `fig_h4_activity_boxplot.png`  
  *Smell density distribution by maintenance intensity*

---

### 5. `part5_visualizations.py`  
**Hypothesis H5: CI/CD Adoption**

**Purpose**  
Assesses whether CI/CD adoption is associated with differences in ML code smell density.

**Required Input**
- `ci_analysis_data.csv`

**Generated Figure**
- `fig_h5_cicd_boxplot.png`  
  *Comparison of ML smell density between CI/CD-enabled and non-enabled projects*

---

### 6. `part6_visualizations.py`  
**Hypothesis H6: ML Domain**

**Purpose**  
Analyzes ML-specific code smells across different ML application domains.

**Required Input**
- `domain_analysis_data.csv`

**Generated Figures**
- `fig_h6_0_boxplot.png`  
  *Distribution of ML smell density across ML domains*
- `fig_h6_1_heatmap.png`  
  *Heatmap showing the prevalence of individual ML smells by domain*

---

## Execution Example

Each script can be run independently:

```bash
python part1_visualizations.py
python part2_visualizations.py
python part3_visualizations.py
python part4_visualizations.py
python part5_visualizations.py
python part6_visualizations.py
