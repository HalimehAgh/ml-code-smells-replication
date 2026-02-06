#!/usr/bin/env python3
"""
Hypothesis Testing H1: LOC vs Pylint Code Smells
================================================
H1.0: Correlation between LOC and smell density
H1.1: Group comparison (small/medium/large projects)

Uses Top 20 Valid Python Smells (excludes imports, style, docstrings)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

class PylintLOCAnalyzer:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir).resolve()
        
        # Pylint result directories
        self.pylint_dirs = {
            'small': 'small_pylint_results',
            'medium': 'medium_pylint_results',
            'large': 'large_pylint_results'
        }
        
        # Top 20 Valid Python Smells (from your test output)
        self.top_20_smells = [
            'unused-argument', 'no-member', 'protected-access', 'no-self-use',
            'too-many-arguments', 'redefined-outer-name', 'too-many-locals',
            'unsubscriptable-object', 'syntax-error', 'too-few-public-methods',
            'attribute-defined-outside-init', 'abstract-method', 'no-else-return',
            'unused-variable', 'fixme', 'too-many-instance-attributes',
            'redefined-builtin', 'arguments-differ', 'logging-fstring-interpolation',
            'no-value-for-parameter'
        ]
        
        self.data = pd.DataFrame()
        self.alpha = 0.05

    def _find_dir(self, dirname):
        """Robustly find directory"""
        candidates = [
            self.base_dir / dirname,
            self.base_dir.parent / dirname,
            self.base_dir / ".." / dirname
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c.resolve()
        return None

    def load_data(self):
        """Load repository data and match with Pylint results"""
        print("="*70)
        print("H1: LOC vs PYLINT CODE SMELLS ANALYSIS")
        print("="*70)
        print("\nStep 1: Loading data...")
        
        all_data = []
        
        for size in ['small', 'medium', 'large']:
            # Load repo CSV
            csv_file = self.base_dir / f'repos_{size}_checked.csv'
            if not csv_file.exists():
                csv_file = self.base_dir / f'repos_{size}.csv'
            if not csv_file.exists():
                print(f"  [!] Warning: CSV for {size} not found")
                continue
            
            repos = pd.read_csv(csv_file)
            
            # Ensure LOC column exists
            if 'Lines of Code' not in repos.columns:
                print(f"  [!] Warning: 'Lines of Code' column missing in {size}")
                continue
            
            repos = repos[repos['Lines of Code'] > 0]  # Filter invalid
            
            # Find Pylint results directory
            pylint_path = self._find_dir(self.pylint_dirs[size])
            if not pylint_path:
                print(f"  [!] Warning: Pylint results not found for {size}")
                continue
            
            print(f"  Processing {size}: {len(repos)} repositories...")
            
            smell_counts = []
            
            for _, repo in repos.iterrows():
                repo_name = repo['GitHub Repo'].strip()
                
                # Matching patterns (handle name variations)
                patterns = [
                    repo_name.replace('/', '_'),  # user_repo
                    repo_name.split('/')[-1],     # Just repo name
                ]
                
                total_smells = 0
                found = False
                
                for pattern in patterns:
                    matches = list(pylint_path.glob(f"*{pattern}*"))
                    if matches:
                        # Take first match
                        repo_folder = matches[0]
                        json_files = list(repo_folder.glob("*.json"))
                        
                        if json_files:
                            try:
                                with open(json_files[0], 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    
                                    # Count ONLY Top 20 valid smells
                                    if isinstance(data, list):
                                        for issue in data:
                                            symbol = issue.get('symbol', '')
                                            if symbol in self.top_20_smells:
                                                total_smells += 1
                            except Exception as e:
                                pass  # Skip problematic files
                        
                        found = True
                        break
                
                if not found:
                    print(f"    [!] No Pylint results for: {repo_name}")
                
                smell_counts.append(total_smells)
            
            repos['total_smells'] = smell_counts
            repos['size_category'] = size
            repos['smell_density'] = (repos['total_smells'] / repos['Lines of Code']) * 1000
            
            all_data.append(repos[['GitHub Repo', 'Lines of Code', 'size_category',
                                  'total_smells', 'smell_density']])
            
            print(f"    ✓ Loaded {size}: {len(repos)} repos")
        
        if not all_data:
            raise ValueError("No data loaded!")
        
        self.data = pd.concat(all_data, ignore_index=True)
        
        # Save for later use
        self.data.to_csv('h1_pylint_analysis_data.csv', index=False)
        print(f"\n  [+] Total repositories: {len(self.data)}")
        print(f"  [+] Saved to: h1_pylint_analysis_data.csv")
        
        return True

    def descriptive_stats(self):
        """Generate descriptive statistics"""
        print("\n" + "="*70)
        print("DESCRIPTIVE STATISTICS")
        print("="*70)
        
        stats_table = self.data.groupby('size_category').agg({
            'Lines of Code': ['count', 'median'],
            'total_smells': ['median'],
            'smell_density': ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        }).round(3)
        
        stats_table.columns = ['N', 'LOC_Median', 'Smells_Median', 'Density_Median', 'Density_Q1', 'Density_Q3']
        stats_table = stats_table.reindex(['small', 'medium', 'large'])
        
        print(stats_table.to_string())
        
        stats_table.to_csv('h1_descriptive_stats.csv')
        print("\n  [+] Saved to: h1_descriptive_stats.csv")
        
        return stats_table

    def correlation_analysis(self):
        """H1.0: Spearman correlation analysis"""
        print("\n" + "="*70)
        print("H1.0: CORRELATION ANALYSIS")
        print("="*70)
        
        results = []
        
        # Test 1: LOC vs Absolute Smell Count
        rho_abs, p_abs = stats.spearmanr(self.data['Lines of Code'], 
                                         self.data['total_smells'])
        
        # Test 2: LOC vs Smell Density (normalized)
        rho_den, p_den = stats.spearmanr(self.data['Lines of Code'], 
                                         self.data['smell_density'])
        
        print(f"\n1. LOC vs Absolute Smell Count:")
        print(f"   Spearman ρ = {rho_abs:.4f}")
        print(f"   p-value    = {p_abs:.2e}")
        print(f"   Strength   = {self._interpret_rho(rho_abs)}")
        print(f"   Decision   = {'REJECT H1.0' if p_abs < self.alpha else 'FAIL TO REJECT H1.0'}")
        
        print(f"\n2. LOC vs Smell Density (normalized):")
        print(f"   Spearman ρ = {rho_den:.4f}")
        print(f"   p-value    = {p_den:.2e}")
        print(f"   Strength   = {self._interpret_rho(rho_den)}")
        print(f"   Decision   = {'REJECT H1.0' if p_den < self.alpha else 'FAIL TO REJECT H1.0'}")
        
        results.append({
            'Relationship': 'LOC vs Absolute Smells',
            'Spearman_rho': rho_abs,
            'p_value': p_abs,
            'Significant': p_abs < self.alpha,
            'Effect_Size': self._interpret_rho(rho_abs)
        })
        
        results.append({
            'Relationship': 'LOC vs Smell Density',
            'Spearman_rho': rho_den,
            'p_value': p_den,
            'Significant': p_den < self.alpha,
            'Effect_Size': self._interpret_rho(rho_den)
        })
        
        # Save results
        pd.DataFrame(results).to_csv('h1_0_correlation_results.csv', index=False)
        print("\n  [+] Saved to: h1_0_correlation_results.csv")
        
        return results

    def group_comparison(self):
        """H1.1: Kruskal-Wallis and pairwise Mann-Whitney U tests"""
        print("\n" + "="*70)
        print("H1.1: GROUP COMPARISON (Small/Medium/Large)")
        print("="*70)
        
        # Prepare groups
        groups = {}
        for size in ['small', 'medium', 'large']:
            groups[size] = self.data[self.data['size_category'] == size]['smell_density']
            print(f"  {size.capitalize()}: n={len(groups[size])}, median={groups[size].median():.3f}")
        
        # Kruskal-Wallis test
        h_stat, p_kw = stats.kruskal(*groups.values())
        
        print(f"\nKruskal-Wallis Test:")
        print(f"  H-statistic = {h_stat:.4f}")
        print(f"  p-value     = {p_kw:.2e}")
        print(f"  Decision    = {'REJECT H1.1' if p_kw < self.alpha else 'FAIL TO REJECT H1.1'}")
        
        if p_kw >= self.alpha:
            print("\n  No significant differences found among groups.")
            return []
        
        print("\n  Significant differences detected! Running post-hoc tests...")
        
        # Post-hoc pairwise comparisons with Bonferroni correction
        pairs = [('small', 'medium'), ('small', 'large'), ('medium', 'large')]
        bonf_alpha = self.alpha / len(pairs)
        
        print(f"\nPost-hoc Mann-Whitney U Tests (Bonferroni α = {bonf_alpha:.4f}):")
        print("-" * 70)
        
        pairwise_results = []
        
        for g1, g2 in pairs:
            u_stat, p_mw = stats.mannwhitneyu(groups[g1], groups[g2], alternative='two-sided')
            delta = self._cliffs_delta(groups[g1], groups[g2])
            significant = p_mw < bonf_alpha
            
            pairwise_results.append({
                'Comparison': f"{g1} vs {g2}",
                'U_statistic': u_stat,
                'p_value': p_mw,
                'Cliffs_delta': delta,
                'Effect_size': self._interpret_delta(delta),
                'Significant': significant,
                'Median_1': groups[g1].median(),
                'Median_2': groups[g2].median()
            })
            
            print(f"  {g1.capitalize()} vs {g2.capitalize()}:")
            print(f"    U = {u_stat:.1f}, p = {p_mw:.2e}, δ = {delta:.3f}")
            print(f"    Effect: {self._interpret_delta(delta)}, Significant: {significant}")
        
        # Save results
        pd.DataFrame(pairwise_results).to_csv('h1_1_pairwise_results.csv', index=False)
        print("\n  [+] Saved to: h1_1_pairwise_results.csv")
        
        return pairwise_results

    def _cliffs_delta(self, x, y):
        """Calculate Cliff's Delta effect size"""
        x, y = np.array(x), np.array(y)
        if len(x) == 0 or len(y) == 0:
            return np.nan
        return np.mean(np.sign(np.subtract.outer(x, y)))

    def _interpret_rho(self, rho):
        """Interpret Spearman's rho"""
        abs_rho = abs(rho)
        if abs_rho < 0.3:
            return "Weak"
        elif abs_rho < 0.7:
            return "Moderate"
        else:
            return "Strong"

    def _interpret_delta(self, delta):
        """Interpret Cliff's Delta"""
        abs_delta = abs(delta)
        if abs_delta < 0.15:
            return "Negligible"
        elif abs_delta < 0.33:
            return "Small"
        elif abs_delta < 0.47:
            return "Medium"
        else:
            return "Large"

    def run_complete_analysis(self):
        """Run complete H1 analysis"""
        print("\nStarting Pylint H1 Analysis...")
        print("Using Top 20 Valid Python Smells (excludes imports/style/docstrings)\n")
        
        if not self.load_data():
            return False
        
        self.descriptive_stats()
        self.correlation_analysis()
        self.group_comparison()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("\nGenerated files:")
        print("  - h1_pylint_analysis_data.csv")
        print("  - h1_descriptive_stats.csv")
        print("  - h1_0_correlation_results.csv")
        print("  - h1_1_pairwise_results.csv")
        
        return True


if __name__ == "__main__":
    analyzer = PylintLOCAnalyzer()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n✓ H1 Analysis completed successfully")
    else:
        print("\n✗ H1 Analysis failed - check data files")