# Code Smell Detection

## Prerequisites and Data Setup

The following CSV files must be present in the same directory as the scripts:

- `repos_small.csv`
- `repos_medium.csv`
- `repos_large.csv`
- `repos_with_domains_final_complete.csv`

---

## CodeSmile Setup and Execution Environment

ML-specific code smell detection is performed using **CodeSmile**, which is part of the **Smell-AI** framework.

- The CodeSmile tool was cloned from the official repository:  
  https://github.com/giammariagiordano/smell_ai

- All CodeSmile-related scripts were executed **directly inside the cloned `smell_ai` directory**.

- A **dedicated Python virtual environment** was created and activated according to the Smell-AI tool’s requirements before running any analysis scripts.

- All dependencies were installed strictly following the Smell-AI documentation to ensure compatibility and reproducibility.

---

## Step 1: Execute ML-Specific Analysis (`run_codesmile.py`)

### Features

- **Cloning**  
  Automatically extracts repository information from the input CSVs and clones repositories into `{size}_repos` directories.

- **Execution**  
  Iterates through the cloned repositories and runs the CodeSmile analysis.

### Output Structure

- `{size}_repos/`  
  Cloned source code repositories.

- `{size}_codesmile_results/`  
  Raw CodeSmile output files for each repository.

- `{size}_codesmile_analysis/analysis_status.csv`  
  Log of successful and failed analysis runs.

---

## Step 2: Execute General Python Analysis (`run_pylint.py`)

Run this script to perform standard static analysis on the same cloned repositories.

### Features

- **Pylint Execution**  
  Runs Pylint on the project directories previously created by the CodeSmile runner.

- **JSON Logging**  
  Results are saved in JSON format for easy programmatic parsing.

### Output Structure

- `{size}_pylint_results/`  
  Subdirectories per repository containing:
  - `.json` Pylint reports
  - `.txt` Pylint reports

---

## Step 3: Process ML Smell Results (`process_codesmile_results.py`)

### Processing Steps

- **Extraction**  
  Parses the `overview.csv` file within each repository’s result folder.

- **Data Aggregation**  
  Extracts raw counts for every detected ML smell per repository.

- **Density Calculation**  
  Computes smell density (total smells per 1,000 Lines of Code) using LOC data from the source CSVs.

### Final Output

- `combined_analysis_data.csv`

---

## Step 4: Process Python Smell Results (`process_pylint_results.py`)

### Two-Pass Logic

1. **Discovery**
   - Scans all repositories to identify the **Top 20 most common functional smells**
   - Automatically excludes noise such as:
     - Style issues (e.g., `invalid-name`)
     - Docstring warnings
     - Environment and import-related errors

### Domain Integration

- Merges results with `repos_with_domains_final_complete.csv`
- Adds domain and contributor characteristics to the dataset

### Final Outputs

- `python_smells_by_repo_with_domains.csv`  
  Primary dataset for general Python smell analysis.

---
