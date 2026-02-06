#!/usr/bin/env python3
"""
Statistical Analysis PART 4: Commit Frequency vs Code Smells
============================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
KEEP_COLUMNS = [
    'Chain_Indexing', 'columns_and_datatype_not_explicitly_set', 'dataframe_conversion_api_misused',
    'in_place_apis_misused', 'matrix_multiplication_api_misused', 'merge_api_parameter_not_explicitly_set',
    'nan_equivalence_comparison_misused', 'unnecessary_iteration', 'gradients_not_cleared_before_backward_propagation',
    'memory_not_freed', 'pytorch_call_method_misused', 'tensor_array_not_used'
]

class CommitAnalyzer:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.df = pd.DataFrame()

    def load_and_process_data(self):
        print("Loading data...")
        try:
            smell_df = pd.read_csv('combined_analysis_data.csv')
        except FileNotFoundError:
            print("Error: combined_analysis_data.csv not found.")
            return False

        cols = [c for c in KEEP_COLUMNS if c in smell_df.columns]
        smell_df['total_smells_filtered'] = smell_df[cols].sum(axis=1)

        meta_frames = []
        for size in ['small', 'medium', 'large']:
            csv_path = self.base_dir / f'repos_{size}.csv'
            if csv_path.exists():
                try:
                    t = pd.read_csv(csv_path)
                    if 'GitHub Repo' in t.columns: t = t.rename(columns={'GitHub Repo': 'repo_name'})
                    req_cols = ['repo_name', 'Commits', 'Created Date', 'Last Commit Date']
                    if all(c in t.columns for c in req_cols):
                        meta_frames.append(t[req_cols])
                except: pass
        
        if not meta_frames: return False
        meta_df = pd.concat(meta_frames, ignore_index=True)
        
        smell_df['repo_name'] = smell_df['repo_name'].str.strip()
        meta_df['repo_name'] = meta_df['repo_name'].str.strip()
        self.df = pd.merge(smell_df, meta_df, on='repo_name', how='inner')
        
        # Filter LOC > 0
        self.df = self.df[self.df['loc'] > 0]
        self.df['smell_density'] = self.df['total_smells_filtered'] / (self.df['loc'] / 1000)
        
        # Load age_days from age_analysis_data.csv
        age_data = pd.read_csv('age_analysis_data.csv')
        if 'GitHub Repo' in age_data.columns:
            age_data = age_data.rename(columns={'GitHub Repo': 'repo_name'})
        age_data['repo_name'] = age_data['repo_name'].str.strip()
        age_data = age_data[['repo_name', 'age_days']]
        self.df = pd.merge(self.df, age_data, on='repo_name', how='inner')
        
        # Calculate Frequency
        self.df['created_dt'] = pd.to_datetime(self.df['Created Date'], utc=True, errors='coerce')
        self.df['last_commit_dt'] = pd.to_datetime(self.df['Last Commit Date'], utc=True, errors='coerce')
        self.df = self.df.dropna(subset=['created_dt', 'last_commit_dt', 'Commits'])
        
        # Duration in months (min 1 month) using age_days
        self.df['duration_months'] = (self.df['age_days'] / 30.44).clip(lower=1)
        self.df['commits_per_month'] = self.df['Commits'] / self.df['duration_months']
        
        # Median Split
        median_freq = self.df['commits_per_month'].median()
        self.df['activity_level'] = self.df['commits_per_month'].apply(
            lambda x: 'Low Activity' if x < median_freq else 'High Activity'
        )
        
        self.df.to_csv('commit_activity_data.csv', index=False)
        return True

    def _cliffs_delta(self, x, y):
        x, y = np.array(x), np.array(y)
        if len(x)==0 or len(y)==0: return np.nan
        return np.mean(np.sign(np.subtract.outer(x, y)))

    def run_analysis(self):
        if not self.load_and_process_data(): return

        # H4.0
        rho, p_corr = stats.spearmanr(self.df['commits_per_month'], self.df['smell_density'])
        
        # H4.1
        low = self.df[self.df['activity_level'] == 'Low Activity']['smell_density']
        high = self.df[self.df['activity_level'] == 'High Activity']['smell_density']
        
        u_stat, p_mw = stats.mannwhitneyu(low, high, alternative='two-sided')
        delta = self._cliffs_delta(low, high)

        # Output
        median_freq = self.df['commits_per_month'].median()
        print("-" * 60)
        print("4. COMMIT FREQUENCY ANALYSIS")
        print("-" * 60)
        print(f"Median Frequency: {median_freq:.2f} commits/month")
        
        print("\n>>> H4.0: CORRELATION")
        print(f"Rho: {rho:.4f} | P: {p_corr:.2e} | {'REJECT H0' if p_corr<0.05 else 'FAIL REJ'}")
        
        print("\n>>> H4.1: GROUP COMPARISON")
        print(f"Low (< {median_freq:.2f}):  N={len(low)}, Median={low.median():.3f}")
        print(f"High (>= {median_freq:.2f}): N={len(high)}, Median={high.median():.3f}")
        print("-" * 30)
        print(f"U-Stat: {u_stat:.1f} | P: {p_mw:.2e} | Delta: {delta:.4f}")
        print(f"Result: {'REJECT H0' if p_mw<0.05 else 'FAIL TO REJECT H0'}")
        print("-" * 60)
        
        # Save results
        res = {
            'rho': rho, 'p_corr': p_corr,
            'u_stat': u_stat, 'p_mw': p_mw, 'delta': delta,
            'median_freq': median_freq
        }
        pd.DataFrame([res]).to_csv('h4_results.csv', index=False)

if __name__ == "__main__":
    CommitAnalyzer().run_analysis()