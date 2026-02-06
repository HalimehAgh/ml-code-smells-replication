#!/usr/bin/env python3
"""
Hypothesis Testing H9.0: Comparison of Contributors Correlations
================================================================
Purpose: Compare team coordination challenges (Contributors vs Density)
         between ML-Specific and General Python code.

Hypothesis:
  H0.9: Correlation(ML_Density, Contributors) == Correlation(Python_Density, Contributors)
  Ha0.9: Correlation(ML_Density, Contributors) != Correlation(Python_Density, Contributors)

Methodology:
1. Load ML Data & Filter: Keep only the 12 Valid ML Smells.
2. Load Python Data: Top 20 General Python Smells.
3. Load Contributors Data.
4. Compare Spearman correlations using Fisher's Z-test.
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
    'Chain_Indexing', 
    'columns_and_datatype_not_explicitly_set', 
    'dataframe_conversion_api_misused', 
    'in_place_apis_misused', 
    'matrix_multiplication_api_misused', 
    'merge_api_parameter_not_explicitly_set',
    'nan_equivalence_comparison_misused', 
    'unnecessary_iteration', 
    'gradients_not_cleared_before_backward_propagation', 
    'memory_not_freed', 
    'pytorch_call_method_misused', 
    'tensor_array_not_used'
]

class H9Analyzer:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir).resolve()
        self.df = pd.DataFrame()

    def _find_file(self, filename):
        candidates = [self.base_dir / filename, self.base_dir.parent / filename, Path(filename)]
        for c in candidates:
            if c.exists() and c.is_file(): return c.resolve()
        return None

    def fisher_z_comparison(self, r1, r2, n):
        """Calculates Fisher's Z for comparing two dependent correlations."""
        z1 = 0.5 * np.log((1 + r1) / (1 - r1))
        z2 = 0.5 * np.log((1 + r2) / (1 - r2))
        se_diff = np.sqrt(1 / (n - 3) + 1 / (n - 3))
        z_stat = (z1 - z2) / se_diff
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        return z_stat, p_val

    def load_data(self):
        print("STEP 1: Loading Data...")
        
        # 1. Load ML Data
        ml_file = self._find_file('combined_analysis_data.csv')
        if not ml_file:
            print("  [!] Error: 'combined_analysis_data.csv' not found.")
            return

        try:
            ml_data = pd.read_csv(ml_file)
            cols_present = [c for c in ML_SMELL_TYPES if c in ml_data.columns]
            ml_data['ml_count_filtered'] = ml_data[cols_present].sum(axis=1)
            ml_data = ml_data[ml_data['loc'] > 0].copy()
            ml_data['ml_density_filtered'] = ml_data['ml_count_filtered'] / (ml_data['loc'] / 1000)
            ml_data['repo_name'] = ml_data['repo_name'].str.strip()
        except Exception as e:
            print(f"  [!] Error reading ML data: {e}")
            return

        # 2. Load Python Data (from paired file)
        py_file = self._find_file('h0_paired_data.csv')
        if not py_file:
            print("  [!] Error: 'h0_paired_data.csv' not found.")
            return
            
        try:
            py_data = pd.read_csv(py_file)
            if 'Repo' in py_data.columns:
                py_data = py_data.rename(columns={'Repo': 'repo_name'})
            py_data['repo_name'] = py_data['repo_name'].str.strip()
        except Exception as e:
            print(f"  [!] Error reading Python data: {e}")
            return

        # 3. Load Contributors Data (Assume it's in age_analysis_data.csv or repos_*.csv)
        # Using age_analysis_data.csv as it was reliable before
        contrib_file = self._find_file('age_analysis_data.csv')
        if not contrib_file:
            print("  [!] Error: 'age_analysis_data.csv' not found.")
            return

        try:
            contrib_data = pd.read_csv(contrib_file)
            if 'GitHub Repo' in contrib_data.columns:
                contrib_data = contrib_data.rename(columns={'GitHub Repo': 'repo_name'})
            contrib_data['repo_name'] = contrib_data['repo_name'].str.strip()
            
            # Ensure 'Contributor Count' exists
            if 'Contributor Count' not in contrib_data.columns:
                 print("  [!] 'Contributor Count' missing in source CSV.")
                 return
        except Exception as e:
            print(f"  [!] Error reading Contributor data: {e}")
            return

        # 4. Merge All
        print("STEP 2: Merging Datasets...")
        merged = pd.merge(ml_data[['repo_name', 'ml_density_filtered']], 
                          py_data[['repo_name', 'Python_Density']], 
                          on='repo_name', how='inner')
        
        self.df = pd.merge(merged, contrib_data[['repo_name', 'Contributor Count']], 
                           on='repo_name', how='inner')
        
        # Filter invalid contributors
        self.df = self.df[self.df['Contributor Count'] > 0]
        
        print(f"  [+] Successfully paired {len(self.df)} repositories.")

    def run_h9_test(self):
        if self.df.empty:
            print("[!] No data available for testing.")
            return

        print("\n" + "="*60)
        print("H9.0: COMPARISON OF CONTRIBUTORS CORRELATIONS")
        print("="*60)
        
        # 1. Spearman Correlations
        rho_ml, p_ml = stats.spearmanr(self.df['Contributor Count'], self.df['ml_density_filtered'])
        rho_py, p_py = stats.spearmanr(self.df['Contributor Count'], self.df['Python_Density'])
        
        print(f"{'Relationship':<35} | {'Spearman rho':<15} | {'p-value'}")
        print("-" * 65)
        print(f"{'Contributors vs ML-Specific':<35} | {rho_ml:<15.4f} | {p_ml:.4e}")
        print(f"{'Contributors vs General Python':<35} | {rho_py:<15.4f} | {p_py:.4e}")
        
        # 2. Fisher's Z-Transformation
        z_stat, p_z = self.fisher_z_comparison(rho_ml, rho_py, len(self.df))
        delta_rho = abs(rho_ml - rho_py)
        
        print("\nFisher's z-transformation Results:")
        print(f"  Difference (Δρ): {delta_rho:.4f}")
        print(f"  z-statistic:     {z_stat:.4f}")
        print(f"  p-value:         {p_z:.4e}")
        
        if delta_rho < 0.1: es = "Negligible"
        elif delta_rho < 0.3: es = "Small"
        elif delta_rho < 0.5: es = "Moderate"
        else: es = "Large"
        print(f"  Effect Size:     {es}")
        
        if p_z < 0.05:
            decision = "REJECT H9.0"
        else:
            decision = "FAIL TO REJECT H9.0"
            
        print(f"  Decision: {decision}")
            
        self._plot(rho_ml, rho_py)

    def _plot(self, rho_ml, rho_py):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Contributors vs ML
        sns.regplot(data=self.df, x='Contributor Count', y='ml_density_filtered', ax=axes[0], 
                    scatter_kws={'alpha':0.5, 's':15}, line_kws={'color':'blue'})
        axes[0].set_title(f'Contributors vs ML-Specific Density\n(rho={rho_ml:.3f})')
        axes[0].set_ylabel('ML Smell Density')
        axes[0].set_xlabel('Contributor Count')
        axes[0].set_xscale('log') # Contributors usually log-normal
        axes[0].set_yscale('log')
        
        # Plot 2: Contributors vs Python
        sns.regplot(data=self.df, x='Contributor Count', y='Python_Density', ax=axes[1], 
                    scatter_kws={'alpha':0.5, 's':15}, line_kws={'color':'purple'})
        axes[1].set_title(f'Contributors vs General Python Density\n(rho={rho_py:.3f})')
        axes[1].set_ylabel('Python Smell Density')
        axes[1].set_xlabel('Contributor Count')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('H9_0_Contributor_Correlation.png')
        print("\n[+] Saved plot: H9_0_Contributor_Correlation.png")

if __name__ == "__main__":
    an = H9Analyzer()
    an.load_data()
    an.run_h9_test()