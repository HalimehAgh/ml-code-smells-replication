#!/usr/bin/env python3
"""
Statistical Analysis PART 2: Project Age vs Code Smells (Median Split)
=====================================================================
Master's Thesis: Code Quality Analysis in ML Repositories

Features:
- Filters for specific 12 code smell types.
- H2.0: Spearman Correlation (Age vs Density).
- H2.1: Mann-Whitney U Test using MEDIAN SPLIT (Young <= Median < Mature).
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
KEEP_COLUMNS = [
    'Chain_Indexing', 'columns_and_datatype_not_explicitly_set', 'dataframe_conversion_api_misused',
    'in_place_apis_misused', 'matrix_multiplication_api_misused', 'merge_api_parameter_not_explicitly_set',
    'nan_equivalence_comparison_misused', 'unnecessary_iteration', 'gradients_not_cleared_before_backward_propagation',
    'memory_not_freed', 'pytorch_call_method_misused', 'tensor_array_not_used'
]

class AgeAnalyzer:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.df = pd.DataFrame()

    def load_and_process_data(self):
        print(f"Loading data...")
        
        # 1. Load Smells
        try:
            smell_df = pd.read_csv('combined_analysis_data.csv')
        except FileNotFoundError:
            print("Error: combined_analysis_data.csv not found.")
            return False

        # Filter cols
        cols = [c for c in KEEP_COLUMNS if c in smell_df.columns]
        smell_df['total_smells_filtered'] = smell_df[cols].sum(axis=1)

        # 2. Load Metadata
        meta_frames = []
        for size in ['small', 'medium', 'large']:
            csv_path = self.base_dir / f'repos_{size}.csv'
            if csv_path.exists():
                try:
                    t = pd.read_csv(csv_path)
                    if 'GitHub Repo' in t.columns: t = t.rename(columns={'GitHub Repo': 'repo_name'})
                    if 'repo_name' in t.columns and 'Created Date' in t.columns:
                        meta_frames.append(t[['repo_name', 'Created Date']])
                except: pass
        
        if not meta_frames: return False
        meta_df = pd.concat(meta_frames, ignore_index=True)
        
        # Load age_days from age_analysis_data.csv
        age_data = pd.read_csv('age_analysis_data.csv')
        if 'GitHub Repo' in age_data.columns:
            age_data = age_data.rename(columns={'GitHub Repo': 'repo_name'})
        age_data['repo_name'] = age_data['repo_name'].str.strip()
        age_data = age_data[['repo_name', 'age_days']]
        
        # 3. Merge
        smell_df['repo_name'] = smell_df['repo_name'].str.strip()
        meta_df['repo_name'] = meta_df['repo_name'].str.strip()
        self.df = pd.merge(smell_df, meta_df, on='repo_name', how='inner')
        self.df = pd.merge(self.df, age_data, on='repo_name', how='inner')
        
        # 4. Age & Density
        self.df['created_dt'] = pd.to_datetime(self.df['Created Date'], utc=True, errors='coerce')
        self.df = self.df.dropna(subset=['created_dt'])
        self.df['age_years'] = self.df['age_days'] / 365.25
        self.df = self.df[self.df['age_days'] > 0]
        
        self.df = self.df[self.df['loc'] > 0]
        self.df['smell_density'] = self.df['total_smells_filtered'] / (self.df['loc'] / 1000)
        
        # 5. MEDIAN SPLIT CLASSIFICATION
        median_age = self.df['age_years'].median()
        print(f"Median Age Calculated: {median_age:.2f} years")
        
        self.df['age_category'] = self.df['age_years'].apply(
            lambda x: 'Young' if x <= median_age else 'Mature'
        )
        
        self.df.to_csv('age_analysis_results_output.csv', index=False)
        return True

    def _cliffs_delta(self, x, y):
        x, y = np.array(x), np.array(y)
        if len(x)==0 or len(y)==0: return np.nan
        return np.mean(np.sign(np.subtract.outer(x, y)))

    def run_analysis(self):
        if not self.load_and_process_data(): return

        # H2.0
        rho, p_corr = stats.spearmanr(self.df['age_days'], self.df['smell_density'])
        
        # H2.1
        young = self.df[self.df['age_category'] == 'Young']['smell_density']
        mature = self.df[self.df['age_category'] == 'Mature']['smell_density']
        
        u_stat, p_mw = stats.mannwhitneyu(young, mature, alternative='two-sided')
        delta = self._cliffs_delta(young, mature)

        # Output
        print("-" * 60)
        print("2. PROJECT AGE/MATURITY - STATISTICAL SUMMARY (MEDIAN SPLIT)")
        print("-" * 60)
        print(f"Median Age: {self.df['age_years'].median():.2f} years")
        
        print("\n>>> H2.0: CORRELATION")
        print(f"Rho: {rho:.4f} | P: {p_corr:.2e} | {'REJECT H0' if p_corr<0.05 else 'FAIL REJ'}")
        
        print("\n>>> H2.1: GROUP COMPARISON")
        print(f"Young (<= Median): N={len(young)}, Median={young.median():.3f}")
        print(f"Mature (> Median): N={len(mature)}, Median={mature.median():.3f}")
        print("-" * 30)
        print(f"U-Stat: {u_stat:.1f} | P: {p_mw:.2e} | Delta: {delta:.4f}")
        print(f"Result: {'REJECT H0' if p_mw<0.05 else 'FAIL TO REJECT H0'}")
        print("-" * 60)
        
        # Save results
        res = {
            'rho': rho, 'p_corr': p_corr,
            'u_stat': u_stat, 'p_mw': p_mw, 'delta': delta,
            'median_age': self.df['age_years'].median()
        }
        pd.DataFrame([res]).to_csv('h2_results.csv', index=False)

if __name__ == "__main__":
    AgeAnalyzer().run_analysis()