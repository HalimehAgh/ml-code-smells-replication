#!/usr/bin/env python3
"""
Statistical Analysis PART 3: Contributors vs Code Smells (Median Split = 53)
===========================================================================
Master's Thesis Analysis

Features:
- Filters for specific 12 code smell types.
- H3.0: Spearman Correlation.
- H3.1: Mann-Whitney U Test with Median Split (Threshold=53).
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
MEDIAN_THRESHOLD = 53

class ContributorAnalyzer:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.df = pd.DataFrame()

    def load_and_process_data(self):
        print("Loading data...")
        
        # 1. Load Smells
        try:
            smell_df = pd.read_csv('combined_analysis_data.csv')
        except FileNotFoundError:
            print("Error: combined_analysis_data.csv not found.")
            return False

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
                    if 'repo_name' in t.columns and 'Contributor Count' in t.columns:
                        meta_frames.append(t[['repo_name', 'Contributor Count']])
                except: pass
        
        if not meta_frames: return False
        meta_df = pd.concat(meta_frames, ignore_index=True)
        
        # 3. Merge
        smell_df['repo_name'] = smell_df['repo_name'].str.strip()
        meta_df['repo_name'] = meta_df['repo_name'].str.strip()
        self.df = pd.merge(smell_df, meta_df, on='repo_name', how='inner')
        
        # 4. Filter & Density
        self.df = self.df[self.df['loc'] > 0]
        self.df['smell_density'] = self.df['total_smells_filtered'] / (self.df['loc'] / 1000)
        self.df = self.df[self.df['Contributor Count'] > 0]
        
        # 5. Categorize (Median Split)
        self.df['team_category'] = self.df['Contributor Count'].apply(
            lambda x: 'Small' if x <= MEDIAN_THRESHOLD else 'Large'
        )
        
        self.df.to_csv('contributors_analysis_data.csv', index=False)
        return True

    def _cliffs_delta(self, x, y):
        x, y = np.array(x), np.array(y)
        if len(x)==0 or len(y)==0: return np.nan
        return np.mean(np.sign(np.subtract.outer(x, y)))

    def run_analysis(self):
        if not self.load_and_process_data(): return

        # H3.0
        rho, p_corr = stats.spearmanr(self.df['Contributor Count'], self.df['smell_density'])
        
        # H3.1
        small = self.df[self.df['team_category'] == 'Small']['smell_density']
        large = self.df[self.df['team_category'] == 'Large']['smell_density']
        
        u_stat, p_mw = stats.mannwhitneyu(small, large, alternative='two-sided')
        delta = self._cliffs_delta(small, large)

        # Output
        print("-" * 60)
        print(f"3. CONTRIBUTORS Analysis (Median Threshold = {MEDIAN_THRESHOLD})")
        print("-" * 60)
        
        print("\n>>> H3.0: CORRELATION")
        print(f"Rho: {rho:.4f} | P: {p_corr:.2e} | {'REJECT H0' if p_corr<0.05 else 'FAIL REJ'}")
        
        print(f"\n>>> H3.1: GROUP COMPARISON")
        print(f"Small (<= {MEDIAN_THRESHOLD}): N={len(small)}, Median={small.median():.3f}")
        print(f"Large (> {MEDIAN_THRESHOLD}):  N={len(large)}, Median={large.median():.3f}")
        print("-" * 30)
        print(f"U-Stat: {u_stat:.1f} | P: {p_mw:.2e} | Delta: {delta:.4f}")
        print(f"Result: {'REJECT H0' if p_mw<0.05 else 'FAIL TO REJECT H0'}")
        print("-" * 60)
        
        # Save results
        res = {
            'rho': rho, 'p_corr': p_corr,
            'u_stat': u_stat, 'p_mw': p_mw, 'delta': delta,
            'threshold': MEDIAN_THRESHOLD
        }
        pd.DataFrame([res]).to_csv('h3_results.csv', index=False)

if __name__ == "__main__":
    ContributorAnalyzer().run_analysis()