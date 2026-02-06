# Data Collection and Preparation

This directory contains the computational pipeline used to update, filter, categorize the NICHE dataset.

## Prerequisites

### System Requirements
* **Python 3.8+**
* **Git:** Required for cloning repositories during metric collection.
* **cloc:** Required for accurate Line of Code (LOC) counting.

### Dependencies
Install the necessary Python libraries:
```bash 
pip install pandas matplotlib seaborn requests
```

### GitHub API Configuration
The update_niche_dataset.py script requires a GitHub Personal Access Token to bypass API rate limits.

Export the token as an environment variable (GITHUB_TOKEN).
To use the GitHub API set your token in the terminal before running:
```bash 
export GITHUB_TOKEN="your_token_here"
```
---

## Execution Pipeline

Execute the scripts in the following sequential order to replicate the dataset construction process.

### 1. Update Metrics
#### **Script:**  update_niche_dataset.py
**Description:** Refreshes repository metrics (stars, commits, lines of code, last update date) and automatically removes repositories with missing commit data.
**Usage:**
```bash 
python update_niche_dataset.py
```
*Note: Use the --resume flag if the process is interrupted.*

### 2. Project Filtering
#### **Script:**  filter_projects.py
**Description:** Applies quantitative exclusion criteria (activity, popularity, maturity, team scale, and workflow complexity) to isolate engineered projects.
**Usage:**
```bash 
python filter_projects.py
```

### 3. Domain Classification
#### **Script:** classify_domains.py
**Description:** Classifies projects into seven distinct machine learning domains using keyword matching and manual verification support.
**Outputs:**
* `repos_with_domains.csv` (Classified data)
* `repos_unclassified_review.csv` (Requires manual classification)
**Usage:**
```bash 
python classify_domains.py
```

### 4. CI/CD Detection
#### **Script:** classify_cicd.py
**Description:** Scans local repositories for configuration files indicative of Continuous Integration and Continuous Deployment (CI/CD) adoption.
**Description:** Generates repos_small.csv, repos_medium.csv, and repos_large.csv
**Usage:**
```bash 
python classify_cicd.py
```
---

## File Structure Overview

* **NICHE.csv:** Original source dataset.
* **niche_updated.csv:** Updated dataset.
* **niche_filtered.csv:** Filtered dataset.
* **repos_with_domains.csv:** Final dataset with domain annotations.
* **repos_with_domains_checked.csv:** Final dataset including CI/CD classification.
* **update_progress.json:** Temporary checkpoint file for long-running update processes.
* **charts/:** Directory containing generated distribution visualizations.
