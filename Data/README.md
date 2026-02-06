# Data Directory

This directory contains all datasets used in the study, organized by processing stage.

## Directory Structure
```
Data/
├── Original_NICHE/           # Original and filtered NICHE datasets
├── Enhanced_Metadata/        # Project features (age, CI/CD, domains)
├── Code_Smell_Results/       # CodeSmile and Pylint outputs
├── Analysis_Inputs/          # Processed data for statistical analysis
├── Figure_Data/              # Data used in paper figures
└── Statistical_Results/      # Complete hypothesis test results
```

---

## File Descriptions

### Original & Updated NICHE/

**Source:** NICHE dataset from Widyasari et al. (2023)

| File | Description | Projects |
|------|-------------|----------|
| `niche_original_572.csv` | Original NICHE dataset (2020) | 572 |
| `niche_updated_565.csv` | Updated metrics (2025): stars, commits, LOC | 565 |
| `niche_297_enriched.csv` | Added: contributors, branches, age | 297 |
| `niche_filtered_279.csv` | **Final dataset** after 5 filtering criteria | 279 |

**Filtering Criteria:**
1. Recent activity (2024+ commits)
2. ≥100 stars
3. ≥100 commits
4. ≥5 contributors
5. ≥2 branches

---

### Enhanced_Metadata/

Additional features extracted for 279 projects:

| File | Features | Description |
|------|----------|-------------|
| `projects_with_age.csv` | age_years | Project age (creation → last commit) |
| `projects_with_cicd.csv` | has_cicd | CI/CD adoption (Yes/No) |
| `repos_manual_classified.csv` | - | 53 manually reviewed ambiguous cases |
| `repos_with_domains_final_complete.csv` | domain | Domain classification (7 categories) |

**7 Domains:**
1. ML Frameworks/Libraries (95, 34.1%)
2. NLP (39, 14.0%)
3. Computer Vision (37, 13.3%)
4. Domain-Specific Applications (33, 11.8%)
5. Data Science/Traditional ML (27, 9.7%)
6. MLOps/Deployment (26, 9.3%)
7. Reinforcement Learning (22, 7.9%)

---

### Code_Smell_Results/

Code smell detection outputs:

| File | Tool | Smells Detected |
|------|------|-----------------|
| `codesmile_results_279.csv` | CodeSmile | 12 ML-specific types |
| `pylint_results_279.csv` | Pylint | Top 20 Python types |
| `combined_smell_densities.csv` | Both | Normalized per 1000 LOC |

**12 ML-Specific Code Smells:**
- CIDX: Chain Indexing
- CDTNE: Columns and DataType Not Explicitly Set
- DCA: Dataframe Conversion API Misused
- IPA: In-Place APIs Misused
- MMA: Matrix Multiplication API Misused
- MAP: Merge API Parameter Not Explicitly Set
- NAN: NaN Equivalence Comparison Misused
- UI: Unnecessary Iteration
- GNC: Gradients Not Cleared
- MNF: Memory Not Freed
- PC: Pytorch Call Method Misused
- TA: TensorArray Not Used

**Top 20 Python Smells:** See paper Table IV

---

### Analysis_Inputs/

Prepared datasets for hypothesis testing:

| File | Used In |
|------|---------|
| `analysis_data.csv` | All RQs - complete dataset |
| `h0_paired_data.csv` | RQ1 (H0) - paired comparisons |
| `h12_data_with_domains.csv` | RQ2/RQ3 (H6), RQ4 (H9) |
| `pylint_comprehensive_data.csv` | RQ3 detailed analysis |

---

### Figure_Data/

Data for paper figures:

| File | Figure |
|------|--------|
| `domain_smell_percentages_ml.csv` | Figure 2 |
| `domain_smell_percentages_python.csv` | Figure 3 |

---

### Statistical_Results/

Complete hypothesis test outputs:

| File | Tests Included |
|------|----------------|
| `rq1_results.txt` | Wilcoxon signed-rank, Cliff's δ |
| `rq2_results.txt` | Spearman's ρ, Mann-Whitney U, Kruskal-Wallis H |
| `rq3_results.txt` | Same as RQ2 |
| `rq4_results.txt` | Fisher's z, bootstrap CI |

---

## Key Variables

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `repo_name` | String | - | GitHub repository name |
| `stars` | Integer | 167-209,443 | GitHub stars |
| `commits` | Integer | 120-188,754 | Total commits |
| `contributors` | Integer | 5-3,231 | Number of contributors |
| `loc` | Integer | 55-2,109,291 | Python lines of code |
| `age_years` | Float | 5.36-16.40 | Project age |
| `has_cicd` | String | Yes/No | CI/CD adoption |
| `domain` | String | 7 categories | Application domain |
| `ml_cs_density` | Float | 0.0-8.5 | ML-CSs per 1000 LOC |
| `python_smell_density` | Float | 12.3-89.7 | Python smells per 1000 LOC |

---

## Size Categories

Based on LOC percentiles:
- **Small:** <10,445 LOC (82 projects, 29.4%)
- **Medium:** 10,445-26,821 LOC (85 projects, 30.5%)
- **Large:** >26,821 LOC (112 projects, 40.1%)

---

## Data Quality

- **Missing values:** None in final 279-project dataset
- **CI/CD validation:** 93.3% automated accuracy
- **Domain validation:** 81.1% automated, 18.9% manual review

---

## References

- NICHE Dataset: Widyasari et al. (2023) - https://doi.org/10.1109/MSR59073.2023.00022
- CodeSmile: https://github.com/AISE-TUDelft/CodeSmile
- For methodology details, see: `Documentation/Supplementary.pdf`

```
