# Comparing ML-Specific and General Python Code Smells Across Project Characteristics

This repository contains the complete replication package for our empirical study analyzing ML-specific code smells versus general Python code smells across 279 open-source machine learning projects.

## Paper Information

**Title:** Comparing ML-Specific and General Python Code Smells Across Project Characteristics

## Study Overview

We examined 279 ML projects from the NICHE dataset containing 132,067 Python files with over 17.9 million lines of code. Our study compares:

- **ML-specific code smells** (12 types) detected using CodeSmile
- **General Python code smells** (top 20 types) detected using Pylint

**Key Findings:**

- ML-CSs occur 41-94× less frequently than general Python smells (1-2% of all issues)
- Commit frequency significantly correlates with ML-specific quality, but age does not
- Standard CI/CD pipelines don't reduce ML-specific technical debt
- Domain-specific patterns require tailored quality strategies

## Repository Structure

```
Replication_Package/
├── Data/                      # All datasets (original, enriched, results)
├── Scripts/                   # Complete analysis pipeline
├── Documentation/             # Methodology and replication guide
└── README.md                  # This file
```

### Data/ Contents

- `Original_NICHE/`: Original and filtered NICHE datasets (572 → 279 projects)
- `Enhanced_Metadata/`: Project features (age, CI/CD, domains)
- `Code_Smell_Results/`: CodeSmile and Pylint outputs
- `Analysis_Inputs/`: Processed data for statistical tests
- `Statistical_Results/`: Complete results for RQ1-RQ4
- `Figure_Data/`: Data used to generate paper figures

### Scripts/ Contents

- `1_Data_Collection_Preparation/`: NICHE updates, filtering, CI/CD & domain classification
- `2_Code_Smell_Detection/`: CodeSmile and Pylint execution scripts
- `3_Statistical_Analysis/`: All hypothesis tests (RQ1-RQ4) with sub-tests
- `4_Visualization/`: Scripts to generate paper figures

### Documentation/ Contents

- Detailed methodology explanations

## Quick Start

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd Replication_Package

# Install dependencies
pip install -r requirements.txt

# Verify data files
ls Data/Original_NICHE/niche_filtered_279.csv
```

### Running the Analysis

**Option 1: View Pre-computed Results**

```bash
# Statistical results
cat Data/Statistical_Results/rq1_results.txt
cat Data/Statistical_Results/rq2_results.txt
cat Data/Statistical_Results/rq3_results.txt
cat Data/Statistical_Results/rq4_results.txt
```

**Option 2: Re-run Statistical Analysis** (approximately 10 minutes)

```bash
cd Scripts/3_Statistical_Analysis

# Run individual research questions
python rq1_analysis.py
python rq2_analysis.py
python rq3_analysis.py
python rq4_analysis.py
```

**Option 3: Full Replication** (10+ hours for smell detection)


## Key Results Files

| Research Question | Results File | Description |
|-------------------|--------------|-------------|
| RQ1 | `Data/Statistical_Results/rq1_results.txt` | Density comparison (ML-CSs vs Python) |
| RQ2 | `Data/Statistical_Results/rq2_results.txt` | ML-CSs correlations with 6 characteristics |
| RQ3 | `Data/Statistical_Results/rq3_results.txt` | Python smells correlations |
| RQ4 | `Data/Statistical_Results/rq4_results.txt` | Differential effect comparisons |

## Main Datasets

| File | Description | 
|------|-------------|
| `niche_filtered_279.csv` | Final dataset: 279 projects with all features |
| `codesmile_results_279.csv` | ML-CSs detected by CodeSmile |
| `pylint_results_279.csv` | Top 20 Python smells from Pylint |
| `combined_smell_densities.csv` | Normalized densities for both smell types |

## Research Questions

**RQ1:** How do the densities of ML-CSs and general Python code smells compare?  
*Finding: ML-CSs are 41-94× less frequent (p < 0.001, large effect)*

**RQ2:** What are the relationships between ML-CSs and project characteristics?  
*Finding: Commit frequency matters (ρ = -0.155, p = 0.010); domain is highly significant*

**RQ3:** What are the relationships between general Python smells and the same characteristics?  
*Finding: No significant relationships; issues are pervasive across all contexts*

**RQ4:** What are the differences in correlation patterns?  
*Finding: Domain rankings are independent (ρ = 0.643, p = 0.119); require tailored strategies*

## Tools Used

- **CodeSmile v1.0** - ML-specific code smell detector (12 smell types)
- **Pylint 2.15.0** - General Python linter (focusing on top 20 functional smells)
- **Python 3.10** with scipy 1.15.3, pandas, numpy, matplotlib
- **Statistical tests:** Spearman's ρ, Mann-Whitney U, Kruskal-Wallis H, Wilcoxon signed-rank


## License

This replication package is released under the MIT License. The NICHE dataset is used under its original license from Widyasari et al. (2023).

## Related Resources

- **Original NICHE Dataset:** Widyasari et al. 2023 [Paper](https://doi.org/10.1109/MSR59073.2023.00022)
- **CodeSmile Tool:** [GitHub Repository](https://github.com/giammariagiordano/smell_ai/tree/main)
- **ML-CSs Catalog:** Zhang et al. 2022 - [Paper](https://doi.org/10.1145/3522664.3528620)

## Contact

For questions about this replication package, please open an issue in this repository.

## Acknowledgments

We thank the maintainers of the NICHE dataset and the developers of CodeSmile and Pylint for making this research possible.
