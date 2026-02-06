Comparing ML-Specific and General Python Code Smells Across Project Characteristics

This repository contains the complete replication package for our empirical study analyzing ML-specific code smells versus general Python code smells across 279 open-source machine learning projects.

📄 Paper Information
Title: Comparing ML-Specific and General Python Code Smells Across Project Characteristics
Conference: ICSME 2026

🎯 Study Overview
We examined 279 ML projects from the NICHE dataset containing 132,067 Python files with over 17.9 million lines of code. Our study compares:

ML-specific code smells (12 types) detected using CodeSmile
General Python code smells (top 20 types) detected using Pylint

Key Findings:

ML-CSs occur 41-94× less frequently than general Python smells (1-2% of all issues)
Commit frequency significantly correlates with ML-specific quality, but age does not
Standard CI/CD pipelines don't reduce ML-specific technical debt
Domain-specific patterns require tailored quality strategies

📁 Repository Structure
Replication_Package/
├── Data/                      # All datasets (original, enriched, results)
├── Scripts/                   # Complete analysis pipeline
├── Documentation/             # Methodology and replication guide
└── README.md                  # This file

Data/ Contents

Original_NICHE/: Original and filtered NICHE datasets (572 → 279 projects)
Enhanced_Metadata/: Project features (age, CI/CD, domains)
Code_Smell_Results/: CodeSmile and Pylint outputs
Analysis_Inputs/: Processed data for statistical tests
Statistical_Results/: Complete results for RQ1-RQ4
Figure_Data/: Data used to generate paper figures

Scripts/ Contents

Data_Collection_Preparation/: NICHE updates, filtering, CI/CD & domain classification
Code_Smell_Detection/: CodeSmile and Pylint execution scripts
Statistical_Analysis/: All hypothesis tests (RQ1-RQ4) with sub-tests
Visualization/: Scripts to generate Figures

Documentation/ Contents

Detailed methodology explanations
