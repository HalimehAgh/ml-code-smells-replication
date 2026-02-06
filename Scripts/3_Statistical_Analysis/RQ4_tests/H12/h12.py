#!/usr/bin/env python3
"""
Hypothesis Testing H12.0: Domain-Specific Pattern Analysis
======================================================================
Purpose: Determine if domain rankings differ between ML-specific and general 
         Python smell densities (i.e., are problematic domains different for 
         ML-specific vs. general Python quality?)

Data Sources:
- repos_with_domains_final_complete.csv: Contains domain classification
- h0_paired_data.csv: Contains filtered ML and Python densities

Methodology:
1. Load domain data and h0 paired data
2. Merge on repository name
3. Calculate median densities by domain
4. Rank domains by each smell type
5. Test correlation between rankings (Spearman's ρ)
6. Identify discordant domains
7. Generate comparative visualizations
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

class H12Analyzer:
    def __init__(self, domain_file='repos_with_domains_final_complete.csv', paired_file='h0_paired_data.csv'):
        self.domain_file = domain_file
        self.paired_file = paired_file
        self.df = pd.DataFrame()
        self.domain_stats = pd.DataFrame()
    
    def _find_file(self, filename):
        """Robustly finds the data file."""
        candidates = [
            Path(filename),
            Path('.').resolve() / filename,
            Path('..').resolve() / filename,
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        return None
    
    def load_data(self):
        """Load and merge domain data with paired densities."""
        print("="*70)
        print("H12.0: DOMAIN-SPECIFIC PATTERN ANALYSIS")
        print("="*70)
        print("\nSTEP 1: Loading Data...")
        
        # Load domain data
        domain_path = self._find_file(self.domain_file)
        if not domain_path:
            print(f"  [!] Error: {self.domain_file} not found.")
            return False
        
        try:
            domain_df = pd.read_csv(domain_path)
            
            # Filter out repos with LOC = 0
            domain_df = domain_df[domain_df['Lines of Code'] > 0].copy()
            
            domain_df['GitHub Repo'] = domain_df['GitHub Repo'].str.strip()
            
            # Find domain column (case-insensitive)
            domain_col = None
            for col in domain_df.columns:
                if col.lower() == 'domain':
                    domain_col = col
                    break
            
            if not domain_col:
                print(f"  [!] Error: 'domain' column not found.")
                return False
            
            print(f"  [+] Loaded domain data: {len(domain_df)} repositories")
            
            # Show domain distribution
            domain_counts = domain_df[domain_col].value_counts()
            print(f"\n  Domain Distribution:")
            for domain, count in domain_counts.items():
                print(f"    {domain:<40}: {count:>3} repos")
            
        except Exception as e:
            print(f"  [!] Error loading domain data: {e}")
            return False
        
        # Load paired data from H0
        paired_path = self._find_file(self.paired_file)
        if not paired_path:
            print(f"  [!] Error: {self.paired_file} not found.")
            print(f"  [!] Make sure you've run h0_combined.py first.")
            return False
        
        try:
            paired_df = pd.read_csv(paired_path)
            paired_df['Repo'] = paired_df['Repo'].str.strip()
            print(f"  [+] Loaded paired data: {len(paired_df)} repositories")
        except Exception as e:
            print(f"  [!] Error loading paired data: {e}")
            return False
        
        # Merge datasets
        domain_subset = domain_df[['GitHub Repo', domain_col]].copy()
        domain_subset = domain_subset.rename(columns={domain_col: 'Domain'})
        
        # Merge on repository name
        merged = pd.merge(
            domain_subset,
            paired_df[['Repo', 'ML_Density', 'Python_Density']],
            left_on='GitHub Repo',
            right_on='Repo',
            how='inner'
        )
        
        # Clean up
        merged = merged.drop(columns=['Repo'])
        merged = merged.rename(columns={'GitHub Repo': 'Repo'})
        
        # Filter out rows with missing domain
        merged = merged[merged['Domain'].notna()].copy()
        
        print(f"  [+] Successfully merged: {len(merged)} repositories with domain info")
        
        # Filter domains with at least 5 repositories for statistical robustness
        domain_counts = merged['Domain'].value_counts()
        valid_domains = domain_counts[domain_counts >= 5].index
        self.df = merged[merged['Domain'].isin(valid_domains)].copy()
        
        print(f"  [+] Filtered to {len(valid_domains)} domains with ≥5 repositories each")
        print(f"  [+] Final dataset: {len(self.df)} repositories")
        
        if len(valid_domains) < 2:
            print("  [!] Error: Need at least 2 domains with ≥5 repos for ranking analysis.")
            return False
        
        return True
    
    def calculate_domain_statistics(self):
        """Calculate median densities by domain."""
        print("\n" + "="*70)
        print("STEP 2: Calculating Domain Statistics")
        print("="*70)
        
        # Calculate median densities by domain
        domain_medians = self.df.groupby('Domain').agg({
            'ML_Density': ['median', 'mean', 'count'],
            'Python_Density': ['median', 'mean']
        }).round(3)
        
        # Flatten column names
        domain_medians.columns = ['_'.join(col).strip() for col in domain_medians.columns.values]
        domain_medians = domain_medians.rename(columns={
            'ML_Density_median': 'ML_Median',
            'ML_Density_mean': 'ML_Mean',
            'ML_Density_count': 'n_repos',
            'Python_Density_median': 'Python_Median',
            'Python_Density_mean': 'Python_Mean'
        })
        
        # Rank domains (1 = highest density = worst quality)
        domain_medians['Rank_ML'] = domain_medians['ML_Median'].rank(ascending=False, method='average')
        domain_medians['Rank_Python'] = domain_medians['Python_Median'].rank(ascending=False, method='average')
        
        # Calculate rank difference
        domain_medians['Rank_Diff'] = domain_medians['Rank_ML'] - domain_medians['Rank_Python']
        domain_medians['Abs_Rank_Diff'] = domain_medians['Rank_Diff'].abs()
        
        self.domain_stats = domain_medians.sort_values('ML_Median', ascending=False)
        
        print(f"\nDomain Rankings (n={len(self.domain_stats)} domains):")
        print("="*80)
        print(f"{'Domain':<35} {'n':<4} {'ML Med':<8} {'Py Med':<8} {'R_ML':<5} {'R_Py':<5} {'Diff'}")
        print("-"*80)
        
        for domain, row in self.domain_stats.iterrows():
            print(f"{domain:<35} {int(row['n_repos']):<4} "
                  f"{row['ML_Median']:<8.3f} {row['Python_Median']:<8.3f} "
                  f"{int(row['Rank_ML']):<5} {int(row['Rank_Python']):<5} "
                  f"{int(row['Rank_Diff']):>+4}")
    
    def run_analysis(self):
        """Perform H12.0 statistical analysis."""
        if self.df.empty or self.domain_stats.empty:
            print("[!] No data available for analysis.")
            return
        
        print("\n" + "="*70)
        print("STEP 3: Ranking Correlation Analysis")
        print("="*70)
        
        # Spearman's rank correlation
        rho, p_value = stats.spearmanr(
            self.domain_stats['Rank_ML'], 
            self.domain_stats['Rank_Python']
        )
        
        print(f"\nSpearman's Rank Correlation:")
        print(f"  ρ (rho):     {rho:.4f}")
        print(f"  p-value:     {p_value:.4e}")
        
        # Interpret correlation
        if abs(rho) < 0.3:
            correlation_strength = "Weak"
        elif abs(rho) < 0.5:
            correlation_strength = "Moderate"
        elif abs(rho) < 0.7:
            correlation_strength = "Strong"
        else:
            correlation_strength = "Very Strong"
        
        if rho > 0:
            correlation_direction = "Positive (Concordant)"
        else:
            correlation_direction = "Negative (Discordant)"
        
        print(f"  Strength:    {correlation_strength}")
        print(f"  Direction:   {correlation_direction}")
        
        print("\n" + "-"*70)
        
        # Decision
        alpha = 0.05
        if p_value < alpha:
            decision = "REJECT H12.0"
            conclusion = "There IS a statistically significant association"
        else:
            decision = "FAIL TO REJECT H12.0"
            conclusion = "There is NO statistically significant association"
        
        print(f"\nDecision (α={alpha}):  {decision}")
        print(f"\nConclusion: {conclusion} between domain rankings")
        print("            for ML-specific vs. general Python smell densities.")
        
        # Interpretation
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        
        if p_value < alpha:
            if rho > 0.5:
                print(f"\nStrong positive correlation (ρ={rho:.3f}) indicates CONCORDANT patterns:")
                print(f"Domains that are problematic for ML-specific smells tend to also be")
                print(f"problematic for general Python smells.")
            elif rho < -0.5:
                print(f"\nStrong negative correlation (ρ={rho:.3f}) indicates DISCORDANT patterns:")
                print(f"Domains that are problematic for ML-specific smells tend to have")
                print(f"BETTER general Python quality, and vice versa.")
            elif abs(rho) < 0.3:
                print(f"\nWeak correlation (ρ={rho:.3f}) indicates INDEPENDENT patterns:")
                print(f"ML-specific and general Python quality issues are largely")
                print(f"domain-specific and independent of each other.")
        else:
            print(f"\nNo significant correlation (ρ={rho:.3f}, p={p_value:.4e}):")
            print(f"Domain rankings for ML-specific and general Python smells are")
            print(f"statistically independent. Different domains face different quality")
            print(f"challenges for each smell type.")
        
        # Identify discordant domains
        print("\n" + "="*70)
        print("STEP 4: Discordant Domain Analysis")
        print("="*70)
        
        # Find top 3 most discordant domains
        discordant = self.domain_stats.nlargest(3, 'Abs_Rank_Diff')
        
        print(f"\nTop 3 Most Discordant Domains (Largest Rank Difference):")
        print("-"*70)
        
        for idx, (domain, row) in enumerate(discordant.iterrows(), 1):
            # Interpret discordance
            if row['Rank_Diff'] < 0:
                # ML rank is lower (better) than Python rank
                pattern = "Better ML quality / Worse Python quality"
                emphasis = "Python"
            else:
                # Python rank is lower (better) than ML rank
                pattern = "Worse ML quality / Better Python quality"
                emphasis = "ML-specific"
            
            print(f"\n{idx}. {domain}")
            print(f"   Pattern:            {pattern}")
            print(f"   ML Median:          {row['ML_Median']:.3f} smells/1k LOC (Rank {int(row['Rank_ML'])})")
            print(f"   Python Median:      {row['Python_Median']:.3f} smells/1k LOC (Rank {int(row['Rank_Python'])})")
            print(f"   Rank Difference:    {int(row['Rank_Diff']):+d}")
            print(f"   Recommendation:     Focus on {emphasis} code quality")
        
        # Calculate domain-specific effect sizes
        self._calculate_domain_effects()
        
        # Save results
        self._save_results(rho, p_value, decision)
        
        # Generate visualizations
        self._plot_domain_comparison()
        self._plot_ranking_scatter()
    
    def _calculate_domain_effects(self):
        """Calculate Cliff's Delta for top discordant domains vs. others."""
        print("\n" + "="*70)
        print("STEP 5: Domain-Specific Effect Sizes")
        print("="*70)
        
        # Get top 3 discordant domains
        top_discordant = self.domain_stats.nlargest(3, 'Abs_Rank_Diff').index
        
        for domain in top_discordant:
            domain_data = self.df[self.df['Domain'] == domain]
            other_data = self.df[self.df['Domain'] != domain]
            
            # Cliff's Delta for ML
            def cliffs_delta(x, y):
                x, y = np.array(x), np.array(y)
                if len(x) == 0 or len(y) == 0: return 0.0
                return np.mean(np.sign(np.subtract.outer(x, y)))
            
            delta_ml = cliffs_delta(domain_data['ML_Density'], other_data['ML_Density'])
            delta_py = cliffs_delta(domain_data['Python_Density'], other_data['Python_Density'])
            
            # Mann-Whitney U test
            u_ml, p_ml = stats.mannwhitneyu(domain_data['ML_Density'], other_data['ML_Density'], alternative='two-sided')
            u_py, p_py = stats.mannwhitneyu(domain_data['Python_Density'], other_data['Python_Density'], alternative='two-sided')
            
            print(f"\n{domain}:")
            print(f"  ML Effect:     δ={delta_ml:>7.3f}, U={u_ml:>8.1f}, p={p_ml:.4e}")
            print(f"  Python Effect: δ={delta_py:>7.3f}, U={u_py:>8.1f}, p={p_py:.4e}")
            
            # Bonferroni correction (α = 0.05 / 6 comparisons = 0.0083)
            bonferroni_alpha = 0.05 / 6
            ml_sig = "✓ Significant" if p_ml < bonferroni_alpha else "  Not significant"
            py_sig = "✓ Significant" if p_py < bonferroni_alpha else "  Not significant"
            print(f"  Bonferroni (α=0.0083): ML {ml_sig}, Py {py_sig}")
    
    def _save_results(self, rho, p_value, decision):
        """Save analysis results to file."""
        results = {
            'test': 'H12.0',
            'n_repos': len(self.df),
            'n_domains': len(self.domain_stats),
            'spearman_rho': rho,
            'p_value': p_value,
            'decision': decision
        }
        
        results_df = pd.DataFrame([results])
        results_df.to_csv('h12_results.csv', index=False)
        print(f"\n[+] Saved statistical results: h12_results.csv")
        
        # Save domain statistics
        self.domain_stats.to_csv('h12_domain_statistics.csv')
        print(f"[+] Saved domain statistics: h12_domain_statistics.csv")
        
        # Save full data
        self.df.to_csv('h12_data_with_domains.csv', index=False)
        print(f"[+] Saved analysis data: h12_data_with_domains.csv")
    
    def _plot_domain_comparison(self):
        """Generate domain comparison bar chart."""
        print("\n[+] Generating visualizations...")
        
        # Prepare data for plotting
        plot_data = self.domain_stats[['ML_Median', 'Python_Median']].reset_index()
        plot_data = plot_data.melt(
            id_vars='Domain',
            value_vars=['ML_Median', 'Python_Median'],
            var_name='Smell Type',
            value_name='Median Density'
        )
        
        # Map column names to readable labels
        plot_data['Smell Type'] = plot_data['Smell Type'].map({
            'ML_Median': 'ML-Specific',
            'Python_Median': 'General Python'
        })
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, max(6, len(self.domain_stats) * 0.5)))
        
        sns.barplot(
            data=plot_data,
            y='Domain',
            x='Median Density',
            hue='Smell Type',
            palette=['#3498db', '#9b59b6'],
            ax=ax
        )
        
        ax.set_xlabel('Median Smell Density (smells/1k LOC)', fontsize=11)
        ax.set_ylabel('Domain', fontsize=11)
        ax.set_title('Code Smell Density by Domain (domains with ≥5 repos)', 
                     fontsize=12, fontweight='bold')
        ax.legend(title='Smell Type', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('H12_0_Domain_Comparison.png', dpi=300, bbox_inches='tight')
        print("[+] Saved plot: H12_0_Domain_Comparison.png")
    
    def _plot_ranking_scatter(self):
        """Generate ranking scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(
            self.domain_stats['Rank_ML'],
            self.domain_stats['Rank_Python'],
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=1
        )
        
        # Add domain labels
        for domain, row in self.domain_stats.iterrows():
            ax.annotate(
                domain,
                (row['Rank_ML'], row['Rank_Python']),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        # Add diagonal line (perfect concordance)
        max_rank = max(self.domain_stats['Rank_ML'].max(), self.domain_stats['Rank_Python'].max())
        ax.plot([1, max_rank], [1, max_rank], 'r--', alpha=0.5, label='Perfect Concordance')
        
        ax.set_xlabel('ML-Specific Rank\n(1 = Highest Density = Worst)', fontsize=11)
        ax.set_ylabel('General Python Rank\n(1 = Highest Density = Worst)', fontsize=11)
        ax.set_title('Domain Rankings: ML-Specific vs. General Python', 
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Invert axes so rank 1 is at top-right (worst quality corner)
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('H12_0_Ranking_Scatter.png', dpi=300, bbox_inches='tight')
        print("[+] Saved plot: H12_0_Ranking_Scatter.png")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)

def main():
    analyzer = H12Analyzer(
        domain_file='repos_with_domains_final_complete.csv',
        paired_file='h0_paired_data.csv'
    )
    
    if analyzer.load_data():
        analyzer.calculate_domain_statistics()
        analyzer.run_analysis()
    else:
        print("\n[!] Analysis aborted due to data loading errors.")
        print("[!] Ensure both repos_with_domains_final_complete.csv and h0_paired_data.csv exist.")

if __name__ == "__main__":
    main()