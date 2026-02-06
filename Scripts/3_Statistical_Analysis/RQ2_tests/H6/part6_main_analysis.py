#!/usr/bin/env python3
"""
Statistical Analysis PART 6: Domain Analysis
============================================
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

class DomainAnalyzer:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.df = pd.DataFrame()

    def load_and_process_data(self):
        print("Loading data...")
        try:
            smell_df = pd.read_csv('combined_analysis_data.csv')
            domain_df = pd.read_csv('repos_with_domains_final_complete.csv')
        except FileNotFoundError:
            print("Error: CSV files not found.")
            return False

        # Clean names
        smell_df['repo_name'] = smell_df['repo_name'].str.strip()
        if 'GitHub Repo' in domain_df.columns:
            domain_df['repo_name'] = domain_df['GitHub Repo'].str.strip()
        
        # Merge
        self.df = pd.merge(smell_df, domain_df[['repo_name', 'domain']], on='repo_name', how='inner')
        
        # Filter Smells
        cols = [c for c in KEEP_COLUMNS if c in self.df.columns]
        self.df['total_smells'] = self.df[cols].sum(axis=1)
        self.df = self.df[self.df['loc'] > 0]
        self.df['smell_density'] = self.df['total_smells'] / (self.df['loc'] / 1000)
        
        # Filter small domains
        counts = self.df['domain'].value_counts()
        valid = counts[counts >= 5].index
        self.df = self.df[self.df['domain'].isin(valid)].copy()
        
        self.df.to_csv('domain_analysis_data.csv', index=False)
        return True

    def _cliffs_delta(self, x, y):
        x, y = np.array(x), np.array(y)
        if len(x)==0 or len(y)==0: return np.nan
        return np.mean(np.sign(np.subtract.outer(x, y)))

    def run_analysis(self):
        if not self.load_and_process_data(): return

        # H6.0 KW
        groups = [g['smell_density'].values for _, g in self.df.groupby('domain')]
        h_stat, p_kw = stats.kruskal(*groups)
        
        # Pairwise
        domains = sorted(self.df['domain'].unique())
        n_comp = (len(domains)*(len(domains)-1))/2
        alpha = 0.05/n_comp
        pairwise = []
        
        if p_kw < 0.05:
            for i in range(len(domains)):
                for j in range(i+1, len(domains)):
                    d1, d2 = domains[i], domains[j]
                    g1 = self.df[self.df['domain']==d1]['smell_density']
                    g2 = self.df[self.df['domain']==d2]['smell_density']
                    u, p = stats.mannwhitneyu(g1, g2)
                    if p < alpha:
                        pairwise.append({'pair': f"{d1} vs {d2}", 'p': p, 'd': self._cliffs_delta(g1, g2)})

        # H6.1 Chi2
        cols = [c for c in KEEP_COLUMNS if c in self.df.columns]
        sums = self.df.groupby('domain')[cols].sum()
        sums = sums.loc[:, (sums != 0).any(axis=0)]
        chi2, p_chi, _, _ = stats.chi2_contingency(sums)
        v = np.sqrt(chi2 / (sums.sum().sum() * (min(sums.shape)-1)))

        # Output
        print("-" * 60)
        print("6. DOMAIN ANALYSIS")
        print("-" * 60)
        print(f"H6.0 KW: H={h_stat:.4f}, p={p_kw:.4e} ({'REJECT' if p_kw<0.05 else 'FAIL'})")
        if pairwise:
            print(f"Significant Pairs (Bonferroni p < {alpha:.4e}):")
            for r in pairwise:
                print(f"  {r['pair']}: p={r['p']:.2e}, delta={r['d']:.3f}")
        
        print(f"\nH6.1 Chi2: X2={chi2:.2f}, p={p_chi:.4e}, V={v:.3f} ({'REJECT' if p_chi<0.05 else 'FAIL'})")
        
        # Save results
        sums.to_csv('domain_smell_counts.csv')

if __name__ == "__main__":
    DomainAnalyzer().run_analysis()