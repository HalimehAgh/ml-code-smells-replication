#!/usr/bin/env python3
"""
Statistical Analysis PART 5: CI/CD Adoption vs Code Smells
==========================================================
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

class CIAnalyzer:
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
            csv_path = self.base_dir / f'repos_{size}_checked.csv'
            if csv_path.exists():
                try:
                    t = pd.read_csv(csv_path)
                    if 'GitHub Repo' in t.columns: t = t.rename(columns={'GitHub Repo': 'repo_name'})
                    # Find CI column
                    ci_col = 'CI_Exists_Checked'
                    if ci_col not in t.columns:
                         # Fallback search
                         for c in t.columns:
                             if 'CI' in c and 'Exist' in c: ci_col = c; break
                    
                    if ci_col in t.columns:
                        temp = t[['repo_name', ci_col]].rename(columns={ci_col: 'CI_Status'})
                        meta_frames.append(temp)
                except: pass
        
        if not meta_frames: 
            print("Error: No metadata loaded.")
            return False
            
        meta_df = pd.concat(meta_frames, ignore_index=True)
        
        smell_df['repo_name'] = smell_df['repo_name'].str.strip()
        meta_df['repo_name'] = meta_df['repo_name'].str.strip()
        self.df = pd.merge(smell_df, meta_df, on='repo_name', how='inner')
        
        # Filter LOC > 0
        self.df = self.df[self.df['loc'] > 0]
        self.df['smell_density'] = self.df['total_smells_filtered'] / (self.df['loc'] / 1000)
        
        self.df.to_csv('ci_analysis_data.csv', index=False)
        return True

    def _cliffs_delta(self, x, y):
        x, y = np.array(x), np.array(y)
        if len(x)==0 or len(y)==0: return np.nan
        return np.mean(np.sign(np.subtract.outer(x, y)))

    def run_analysis(self):
        if not self.load_and_process_data(): return

        # H5.0
        ci_yes = self.df[self.df['CI_Status'] == 'Yes']['smell_density']
        ci_no = self.df[self.df['CI_Status'] == 'No']['smell_density']
        
        u_stat, p_mw = stats.mannwhitneyu(ci_yes, ci_no, alternative='two-sided')
        delta = self._cliffs_delta(ci_yes, ci_no)

        # Output
        print("-" * 60)
        print("5. CI/CD ADOPTION ANALYSIS")
        print("-" * 60)
        
        print("\n>>> H5.0: GROUP COMPARISON")
        print(f"CI (Yes):     N={len(ci_yes)}, Median={ci_yes.median():.3f}")
        print(f"Non-CI (No):  N={len(ci_no)}, Median={ci_no.median():.3f}")
        print("-" * 30)
        print(f"U-Stat: {u_stat:.1f} | P: {p_mw:.2e} | Delta: {delta:.4f}")
        print(f"Result: {'REJECT H0' if p_mw<0.05 else 'FAIL TO REJECT H0'}")
        print("-" * 60)
        
        # Save results
        res = {
            'u_stat': u_stat, 'p_mw': p_mw, 'delta': delta
        }
        pd.DataFrame([res]).to_csv('h5_results.csv', index=False)

if __name__ == "__main__":
    CIAnalyzer().run_analysis()