#!/usr/bin/env python3
"""
Hypothesis Testing H8.0: Comparison of Age/Maturity Correlations
================================================================
Purpose: Determine if technical debt accumulation (correlation with Age)
         differs between ML-specific issues and General Python issues.

Hypothesis:
  H0.8: Correlation(ML_Density, Age) == Correlation(Python_Density, Age)
  Ha0.8: Correlation(ML_Density, Age) != Correlation(Python_Density, Age)

Methodology:
1. Load 'age_analysis_data.csv' (Source of Age).
2. Load 'combined_analysis_data.csv' (Source of ML Density).
3. Filter ML data to keep only the 12 valid ML smells.
4. Load 'h0_paired_data.csv' (Source of Python Density).
5. Compare Spearman correlations using Fisher's Z-test.
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

# --- CONFIGURATION ---
# The 12 ML smell types to INCLUDE (Excluding the 4 specific ones)
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

def fisher_z_comparison(r1, r2, n):
    """Calculates Fisher's Z for comparing two dependent correlations."""
    # Convert r to z
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    
    # Standard Error (Approximation for independent samples often used)
    se_diff = np.sqrt(1 / (n - 3) + 1 / (n - 3))
    
    # Test Statistic
    z_stat = (z1 - z2) / se_diff
    
    # P-value (two-tailed)
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_val

class H8Analyzer:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir).resolve()
        self.df = pd.DataFrame()

    def _find_file(self, filename):
        candidates = [self.base_dir / filename, self.base_dir.parent / filename, Path(filename)]
        for c in candidates:
            if c.exists() and c.is_file(): return c.resolve()
        return None

    def load_data(self):
        print("STEP 1: Loading Data...")
        
        # 1. Load ML Data (to filter smells)
        ml_file = self._find_file('combined_analysis_data.csv')
        if not ml_file:
            print("  [!] Error: 'combined_analysis_data.csv' not found.")
            return

        try:
            ml_data = pd.read_csv(ml_file)
            # Recalculate ML Density using ONLY the 12 types
            cols_present = [c for c in ML_SMELL_TYPES if c in ml_data.columns]
            ml_data['ml_count_filtered'] = ml_data[cols_present].sum(axis=1)
            # Filter valid LOC
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

        # 3. Load Age Data
        age_file = self._find_file('age_analysis_data.csv')
        if not age_file:
            print("  [!] Error: 'age_analysis_data.csv' not found.")
            return

        try:
            age_data = pd.read_csv(age_file)
            if 'GitHub Repo' in age_data.columns:
                age_data = age_data.rename(columns={'GitHub Repo': 'repo_name'})
            age_data['repo_name'] = age_data['repo_name'].str.strip()
        except Exception as e:
            print(f"  [!] Error reading Age data: {e}")
            return

        # 4. Merge All
        print("STEP 2: Merging Datasets...")
        merged = pd.merge(ml_data[['repo_name', 'ml_density_filtered']], 
                          py_data[['repo_name', 'Python_Density']], 
                          on='repo_name', how='inner')
        
        self.df = pd.merge(merged, age_data[['repo_name', 'age_days']], on='repo_name', how='inner')
        print(f"  [+] Successfully paired {len(self.df)} repositories.")

    def run_h8_test(self):
        if self.df.empty:
            print("[!] No data available for testing.")
            return

        print("\n" + "="*60)
        print("H8.0: COMPARISON OF AGE CORRELATIONS (Maturity vs Quality)")
        print("="*60)
        
        # 1. Spearman Correlations
        rho_ml, p_ml = stats.spearmanr(self.df['age_days'], self.df['ml_density_filtered'])
        rho_py, p_py = stats.spearmanr(self.df['age_days'], self.df['Python_Density'])
        
        print(f"{'Relationship':<30} | {'Spearman rho':<15} | {'p-value'}")
        print("-" * 60)
        print(f"{'Age vs ML-Specific':<30} | {rho_ml:<15.4f} | {p_ml:.4e}")
        print(f"{'Age vs General Python':<30} | {rho_py:<15.4f} | {p_py:.4e}")
        
        # 2. Fisher's Z-Transformation
        z_stat, p_z = fisher_z_comparison(rho_ml, rho_py, len(self.df))
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
            decision = "REJECT H8.0"
            conclusion = "Significant difference in how technical debt accumulates over time."
        else:
            decision = "FAIL TO REJECT H8.0"
            conclusion = "No significant difference in age correlations."
            
        print(f"  Decision: {decision}")
        print(f"  Conclusion: {conclusion}")
            
        self._plot(rho_ml, rho_py)

    def _plot(self, rho_ml, rho_py):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Age vs ML
        sns.regplot(data=self.df, x='age_days', y='ml_density_filtered', ax=axes[0], 
                    scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
        axes[0].set_title(f'Age vs ML-Specific Density\n(rho={rho_ml:.3f})')
        axes[0].set_ylabel('ML Smell Density')
        axes[0].set_xlabel('Project Age (Days)')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        
        # Plot 2: Age vs Python
        sns.regplot(data=self.df, x='age_days', y='Python_Density', ax=axes[1], 
                    scatter_kws={'alpha':0.5}, line_kws={'color':'purple'})
        axes[1].set_title(f'Age vs General Python Density\n(rho={rho_py:.3f})')
        axes[1].set_ylabel('Python Smell Density')
        axes[1].set_xlabel('Project Age (Days)')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('H8_0_Age_Correlation.png')
        print("\n[+] Saved plot: H8_0_Age_Correlation.png")

if __name__ == "__main__":
    an = H8Analyzer()
    an.load_data()
    an.run_h8_test()