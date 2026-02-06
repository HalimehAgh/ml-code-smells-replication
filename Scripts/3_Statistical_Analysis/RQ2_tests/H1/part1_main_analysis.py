#!/usr/bin/env python3
"""
Statistical Analysis: LOC vs Code Smells (Filtered)
===================================================
Master's Thesis: Code Quality Analysis in ML Repositories

Updates:
- Filters for specific 12 code smell types (ignores 4 types).
- H1.0: Spearman correlation (LOC vs Count, LOC vs Density).
- H1.1: Kruskal-Wallis + MWU (Bonferroni) + Cliff's Delta.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# The 12 columns to KEEP for analysis
KEEP_COLUMNS = [
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

class LOCAnalyzer:
    def __init__(self, input_file='combined_analysis_data.csv'):
        self.input_file = Path(input_file)
        self.data = pd.DataFrame()
        self.alpha = 0.05
    
    def load_and_process_data(self):
        """Load data and recalculate totals based on filtered columns"""
        print(f"Loading data from {self.input_file}...")
        
        if not self.input_file.exists():
            print(f"Error: {self.input_file} not found.")
            return False
            
        df = pd.read_csv(self.input_file)
        
        # Ensure 'loc' column exists (normalize names if needed)
        if 'loc' not in df.columns and 'Lines of Code' in df.columns:
            df = df.rename(columns={'Lines of Code': 'loc'})
            
        # Filter valid LOC
        df = df[df['loc'] > 0].copy()
        
        # Recalculate Total Smells using ONLY the 12 kept columns
        print("Recalculating totals for the 12 filtered smell types...")
        missing_cols = [c for c in KEEP_COLUMNS if c not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in CSV: {missing_cols}")
            # Fill missing with 0 to proceed safely
            for c in missing_cols:
                df[c] = 0
                
        df['total_smells'] = df[KEEP_COLUMNS].sum(axis=1)
        
        # Recalculate Density (per 1K LOC)
        df['smell_density'] = df['total_smells'] / (df['loc'] / 1000)
        
        self.data = df
        
        # Save processed data for visualization
        self.data.to_csv('analysis_data.csv', index=False)
        print(f"Data processed: {len(self.data)} repositories.")
        print("Saved processed data to 'analysis_data.csv'")
        return True
    
    def descriptive_stats(self):
        """Generate descriptive statistics table"""
        stats_table = self.data.groupby('size_category').agg({
            'loc': ['count', 'median'],
            'total_smells': ['median'],
            'smell_density': ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        }).round(3)
        
        stats_table.columns = ['N', 'LOC_Median', 'Smells_Median', 'Density_Median', 'Density_Q1', 'Density_Q3']
        try:
            stats_table = stats_table.reindex(['small', 'medium', 'large'])
        except:
            pass # In case categories are different
            
        print("\nDescriptive Statistics (Filtered Data):")
        print(stats_table)
        
        stats_table.to_csv('descriptive_stats.csv')
        return stats_table
    
    def correlation_analysis(self):
        """H1.0: Spearman correlation analysis"""
        print("\n" + "="*60)
        print("H1.0: CORRELATION ANALYSIS (Project Size vs. Smells)")
        print("="*60)
        
        results = []
        
        # Test 1: LOC vs Absolute Smell Count
        rho_abs, p_abs = stats.spearmanr(self.data['loc'], self.data['total_smells'])
        
        # Test 2: LOC vs Smell Density
        rho_den, p_den = stats.spearmanr(self.data['loc'], self.data['smell_density'])
        
        # Helper to decide
        def decision(p): return "REJECT H1.0" if p < self.alpha else "FAIL TO REJECT H1.0"
        
        results.append({
            'Relationship': 'LOC vs Absolute Smells',
            'Spearman_rho': rho_abs,
            'p_value': p_abs,
            'Interpretation': self._interpret_rho(rho_abs),
            'Decision': decision(p_abs)
        })
        
        results.append({
            'Relationship': 'LOC vs Smell Density',
            'Spearman_rho': rho_den,
            'p_value': p_den,
            'Interpretation': self._interpret_rho(rho_den),
            'Decision': decision(p_den)
        })
        
        # Print Report
        print(f"1. LOC vs Absolute Smell Count:")
        print(f"   rho = {rho_abs:.4f}, p = {p_abs:.4e} ({results[0]['Interpretation']})")
        print(f"   -> {results[0]['Decision']}")
        
        print(f"\n2. LOC vs Smell Density (Normalized):")
        print(f"   rho = {rho_den:.4f}, p = {p_den:.4e} ({results[1]['Interpretation']})")
        print(f"   -> {results[1]['Decision']}")
        
        # Save results
        pd.DataFrame(results).to_csv('h1_0_correlation_results.csv', index=False)
        return results
    
    def group_comparison(self):
        """H1.1: Kruskal-Wallis and pairwise tests"""
        print("\n" + "="*60)
        print("H1.1: SIZE GROUP COMPARISON (Small vs Medium vs Large)")
        print("="*60)
        
        # Prepare group data
        groups = {}
        for size in ['small', 'medium', 'large']:
            subset = self.data[self.data['size_category'] == size]['smell_density']
            groups[size] = subset
            print(f"{size.title()}: n={len(subset)}, Median Density={subset.median():.3f}")
        
        # Kruskal-Wallis test
        h_stat, p_kw = stats.kruskal(groups['small'], groups['medium'], groups['large'])
        
        print(f"\nKruskal-Wallis Test:")
        print(f"H = {h_stat:.4f}, p = {p_kw:.4e}")
        
        if p_kw >= self.alpha:
            print("-> FAIL TO REJECT H1.1 (No significant difference among groups)")
        else:
            print("-> REJECT H1.1 (Significant difference found)")
        
        # Post-hoc pairwise comparisons (Bonferroni)
        pairs = [('small', 'medium'), ('small', 'large'), ('medium', 'large')]
        bonf_alpha = 0.05 / 3  # 0.017
        
        print(f"\nPost-hoc Mann-Whitney U tests (Bonferroni α = {bonf_alpha:.4f}):")
        
        pairwise_results = []
        
        for g1, g2 in pairs:
            u_stat, p_mw = stats.mannwhitneyu(groups[g1], groups[g2], alternative='two-sided')
            
            # Bonferroni corrected p-value for reporting
            p_adj = min(p_mw * 3, 1.0)
            
            delta = self._cliffs_delta(groups[g1], groups[g2])
            interp_delta = self._interpret_delta(delta)
            
            # Decision based on corrected alpha
            sig = "Significant" if p_mw < bonf_alpha else "Not Significant"
            
            pairwise_results.append({
                'Comparison': f"{g1} vs {g2}",
                'U_statistic': u_stat,
                'p_unadj': p_mw,
                'p_adj': p_adj,
                'Cliffs_delta': delta,
                'Effect_Size': interp_delta,
                'Result': sig
            })
            
            print(f"{g1} vs {g2}:")
            print(f"  U={u_stat:.1f}, p_adj={p_adj:.4e}, Delta={delta:.3f} ({interp_delta})")
            print(f"  -> {sig}")
        
        # Save pairwise results
        pd.DataFrame(pairwise_results).to_csv('h1_1_pairwise_results.csv', index=False)
        return pairwise_results
    
    def _cliffs_delta(self, x, y):
        """Calculate Cliff's Delta"""
        x, y = np.array(x), np.array(y)
        if len(x) == 0 or len(y) == 0: return np.nan
        
        # Efficient vectorized calculation
        # Delta = ( #(x>y) - #(x<y) ) / (n_x * n_y)
        differences = np.subtract.outer(x, y)
        return np.mean(np.sign(differences))
    
    def _interpret_rho(self, rho):
        """Interpret Spearman correlation strength"""
        abs_rho = abs(rho)
        if abs_rho < 0.3: return "Weak"
        elif abs_rho < 0.7: return "Moderate"
        else: return "Strong"
    
    def _interpret_delta(self, delta):
        """Interpret Cliff's Delta effect size"""
        d = abs(delta)
        if d < 0.15: return "Negligible"
        elif d < 0.33: return "Small"
        elif d < 0.47: return "Medium"
        else: return "Large"
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        if not self.load_and_process_data():
            return False
        
        self.descriptive_stats()
        self.correlation_analysis()
        self.group_comparison()
        
        print(f"\nAnalysis complete. Outputs:")
        print("- analysis_data.csv")
        print("- descriptive_stats.csv")
        print("- h1_0_correlation_results.csv")
        print("- h1_1_pairwise_results.csv")
        return True

if __name__ == "__main__":
    analyzer = LOCAnalyzer()
    analyzer.run_analysis()