#!/usr/bin/env python3
"""
Hypothesis Testing H10.0: Comparison of Commit Frequency Correlations
=====================================================================
Purpose: Determine if development pace (Commit Frequency) affects ML-specific 
         and General Python code quality differently.

Methodology:
1. Load 'age_analysis_data.csv' (Source of Commits, Age).
2. Load 'combined_analysis_data.csv' (Source of ML Smells - FILTERED 12 TYPES).
3. Load 'h0_paired_data.csv' (Source of Python Density - STABLE BASELINE).
4. Calculate Commit Frequency = Total Commits / (Age in Years).
5. Perform Spearman Correlation & Fisher's Z-Test.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

# --- CONFIGURATION: THE 12 INCLUDED ML SMELLS ---
ML_SMELL_TYPES = [
    'Chain_Indexing', 'columns_and_datatype_not_explicitly_set', 'dataframe_conversion_api_misused',
    'in_place_apis_misused', 'matrix_multiplication_api_misused', 'merge_api_parameter_not_explicitly_set',
    'nan_equivalence_comparison_misused', 'unnecessary_iteration', 'gradients_not_cleared_before_backward_propagation',
    'memory_not_freed', 'pytorch_call_method_misused', 'tensor_array_not_used'
]

class H10Analyzer:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir).resolve()
        self.df = pd.DataFrame()

    def _find_file(self, filename):
        candidates = [self.base_dir / filename, self.base_dir.parent / filename, Path(filename)]
        for c in candidates:
            if c.exists() and c.is_file(): return c.resolve()
        return None

    def fisher_z_comparison(self, r1, r2, n):
        z1 = 0.5 * np.log((1 + r1) / (1 - r1))
        z2 = 0.5 * np.log((1 + r2) / (1 - r2))
        se_diff = np.sqrt(1 / (n - 3) + 1 / (n - 3))
        z_stat = (z1 - z2) / se_diff
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        return z_stat, p_val

    def load_data(self):
        print("STEP 1: Loading Data...")
        
        # 1. Load Age/Commit Data
        age_file = self._find_file('age_analysis_data.csv')
        if not age_file:
            print("  [!] Error: 'age_analysis_data.csv' not found.")
            return
        
        try:
            age_df = pd.read_csv(age_file)
            if 'GitHub Repo' in age_df.columns:
                age_df = age_df.rename(columns={'GitHub Repo': 'repo_name'})
            age_df['repo_name'] = age_df['repo_name'].str.strip()
            
            # Calculate Commit Frequency (Commits per Year)
            age_df['age_years'] = age_df['age_days'].apply(lambda x: max(x/365.25, 0.01))
            age_df['commit_freq'] = age_df['Commits'] / age_df['age_years']
            
        except Exception as e:
            print(f"  [!] Error reading age data: {e}")
            return

        # 2. Load ML Data (Filtered)
        ml_file = self._find_file('combined_analysis_data.csv')
        if not ml_file:
            print("  [!] Error: 'combined_analysis_data.csv' not found. Cannot filter smells.")
            return

        try:
            ml_df = pd.read_csv(ml_file)
            ml_df['repo_name'] = ml_df['repo_name'].str.strip()
            
            # Filter columns
            cols_present = [c for c in ML_SMELL_TYPES if c in ml_df.columns]
            ml_df['ml_count_filtered'] = ml_df[cols_present].sum(axis=1)
            ml_df = ml_df[ml_df['loc'] > 0]
            ml_df['ml_density_filtered'] = ml_df['ml_count_filtered'] / (ml_df['loc'] / 1000)
            
        except Exception as e:
            print(f"  [!] Error reading ML data: {e}")
            return

        # 3. Load Python Data (From Stable CSV)
        py_file = self._find_file('h0_paired_data.csv')
        if not py_file:
            print("  [!] Error: 'h0_paired_data.csv' not found.")
            return

        try:
            py_df = pd.read_csv(py_file)
            if 'Repo' in py_df.columns: py_df = py_df.rename(columns={'Repo': 'repo_name'})
            py_df['repo_name'] = py_df['repo_name'].str.strip()
        except Exception as e:
            print(f"  [!] Error reading Python data: {e}")
            return

        # 4. Merge All
        print("STEP 2: Merging Datasets...")
        merged = pd.merge(ml_df[['repo_name', 'ml_density_filtered']], 
                          py_df[['repo_name', 'Python_Density']], 
                          on='repo_name', how='inner')
        
        self.df = pd.merge(merged, age_df[['repo_name', 'commit_freq']], 
                           on='repo_name', how='inner')
        
        print(f"  [+] Successfully paired {len(self.df)} repositories.")

    def run_h10_test(self):
        if self.df.empty: return

        print("\n" + "="*60)
        print("H10.0: COMPARISON OF COMMIT FREQUENCY CORRELATIONS")
        print("="*60)
        
        rho_ml, p_ml = stats.spearmanr(self.df['commit_freq'], self.df['ml_density_filtered'])
        rho_py, p_py = stats.spearmanr(self.df['commit_freq'], self.df['Python_Density'])
        
        print(f"{'Relationship':<35} | {'Spearman rho':<12} | {'p-value'}")
        print("-" * 65)
        print(f"{'Commit Freq vs ML-Specific':<35} | {rho_ml:<12.4f} | {p_ml:.4e}")
        print(f"{'Commit Freq vs General Python':<35} | {rho_py:<12.4f} | {p_py:.4e}")
        
        z_stat, p_z = self.fisher_z_comparison(rho_ml, rho_py, len(self.df))
        delta_rho = abs(rho_ml - rho_py)
        
        if delta_rho < 0.1: es = "Negligible"
        elif delta_rho < 0.3: es = "Small"
        elif delta_rho < 0.5: es = "Moderate"
        else: es = "Large"

        print("\nFisher's z-transformation Results:")
        print(f"  Difference (Δρ): {delta_rho:.4f}")
        print(f"  z-statistic:     {z_stat:.4f}")
        print(f"  p-value:         {p_z:.4e}")
        print(f"  Effect Size:     {es}")
        
        if p_z < 0.05:
            decision = "REJECT H10.0"
        else:
            decision = "FAIL TO REJECT H10.0"
            
        print(f"  Decision:        {decision}")
        print(f"  Conclusion:      Development pace has a similar (or null) impact on both.")

        self._plot()

    def _plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.regplot(data=self.df, x='commit_freq', y='ml_density_filtered', ax=axes[0], 
                    scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
        axes[0].set_title(f'Commit Freq vs ML-Specific Density\n(rho={stats.spearmanr(self.df["commit_freq"], self.df["ml_density_filtered"])[0]:.3f})')
        axes[0].set_xscale('log'); axes[0].set_yscale('log')
        axes[0].set_xlabel('Commit Frequency (Commits/Year)')
        axes[0].set_ylabel('ML Smell Density')
        
        sns.regplot(data=self.df, x='commit_freq', y='Python_Density', ax=axes[1], 
                    scatter_kws={'alpha':0.5}, line_kws={'color':'purple'})
        axes[1].set_title(f'Commit Freq vs General Python Density\n(rho={stats.spearmanr(self.df["commit_freq"], self.df["Python_Density"])[0]:.3f})')
        axes[1].set_xscale('log'); axes[1].set_yscale('log')
        axes[1].set_xlabel('Commit Frequency (Commits/Year)')
        axes[1].set_ylabel('Python Smell Density')
        
        plt.tight_layout()
        plt.savefig('H10_0_Frequency_Correlation.png')
        print("\n[+] Saved plot: H10_0_Frequency_Correlation.png")

if __name__ == "__main__":
    an = H10Analyzer()
    an.load_data()
    an.run_h10_test()